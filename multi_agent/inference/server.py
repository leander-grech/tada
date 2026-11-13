"""Minimal FastAPI server for running inference on trained PPO models."""

import logging
import os
import secrets
import shutil
import threading
import zipfile
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager

import ray
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from flight_simulator import World
from starlette.requests import Request
from starlette.responses import JSONResponse

from config.config import Config
from inference.checkpoint_manager import (
    RLLIB_MARKER,
    MultiCheckpointManager,
    checkpoint_name_from_upload,
    extract_checkpoint_zip,
    validate_checkpoint,
)
from inference.engine import InferenceEngine
from inference.logging_utils import setup_logging
from inference.server_models import (
    ActionRecord,
    Airport,
    AirportCheckpointInfo,
    AirportProgressInfo,
    CheckpointListResponse,
    InferenceRequest,
    InferenceResponse,
    ModelConfigInfo,
    SetCheckpointRequest,
    Suggestion,
    SuggestionsRequest,
    TaskStartResponse,
    TaskStatusResponse,
    UploadCheckpointResponse,
)
from inference.service import (
    InferenceCancelled,
    NoValidSolutionError,
    run_inference,
    run_inference_candidate,
)
from inference.snapshot_converter import SnapshotConverter
from inference.suggestion_converter import convert_actions_to_suggestions
from inference.task_store import TaskStatus, TaskStore

logger = logging.getLogger(__name__)

MAX_CHECKPOINT_ZIP_SIZE = 30 * 1024 * 1024  # 30 MB
MAX_CHECKPOINT_COUNT = 20


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
_AIRPORTS = [a.value for a in Airport]

# ~75 MiB per Ray inference engine
_RAY_OBJECT_STORE_PER_ENGINE = 78_643_200


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize Ray + MultiCheckpointManager + SnapshotConverter on startup."""
    checkpoints_dir = os.environ.get("CHECKPOINTS_DIR")
    if not checkpoints_dir:
        raise RuntimeError("CHECKPOINTS_DIR environment variable is required")

    manager: MultiCheckpointManager | None = None
    ray_started = False

    try:
        ray.init(
            ignore_reinit_error=True,
            runtime_env={},
            logging_level="ERROR",
            log_to_driver=False,
            object_store_memory=_RAY_OBJECT_STORE_PER_ENGINE * len(_AIRPORTS),
        )
        ray_started = True

        manager = MultiCheckpointManager(
            checkpoints_dir,
            airports=_AIRPORTS,
            max_checkpoints=MAX_CHECKPOINT_COUNT,
        )
        for airport, available in manager.discover_all().items():
            logger.info("[%s] Discovered %d checkpoints: %s", airport, len(available), available)

            mgr = manager.get(airport)
            default_name = mgr.select_default()
            if default_name is not None:
                logger.info("[%s] Loading default checkpoint: %s", airport, default_name)
                try:
                    mgr.load(default_name)
                except (ValueError, RuntimeError) as exc:
                    logger.warning(
                        "[%s] Default checkpoint %r failed to load: %s",
                        airport,
                        default_name,
                        exc,
                    )

        _app.state.multi_manager = manager
        _app.state.converter = SnapshotConverter()

        yield
    finally:
        if manager is not None:
            manager.shutdown_all()
        if ray_started and ray.is_initialized():
            ray.shutdown()


app = FastAPI(title="ATC Inference API", lifespan=lifespan)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err["loc"])
        errors.append(f"{loc}: {err['msg']}")
    detail = "; ".join(errors)
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, detail)
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(NoValidSolutionError)
async def no_valid_solution_handler(request: Request, exc: NoValidSolutionError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc), "candidates": exc.candidates})


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _is_api_key_enabled() -> bool:
    """Return True when API-key enforcement is enabled by env."""
    value = os.environ.get("INFERENCE_API_KEY_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Require a valid API key when auth is enabled."""
    if not _is_api_key_enabled():
        return

    configured_key = os.environ.get("INFERENCE_API_KEY", "")
    if not configured_key:
        raise HTTPException(
            status_code=500,
            detail="API key auth is enabled but INFERENCE_API_KEY is not configured.",
        )

    if not api_key or not secrets.compare_digest(api_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_engine(airport: str) -> InferenceEngine:
    """Return the active inference engine for the given airport."""
    manager = app.state.multi_manager.get(airport)
    return manager.engine


_airport_locks = {a: threading.Lock() for a in _AIRPORTS}
_INFERENCE_LOCK_TIMEOUT_S = float(os.environ.get("INFERENCE_LOCK_TIMEOUT_S", "60"))

_task_store = TaskStore()


@contextmanager
def _with_inference_lock(airport: str):
    """Acquire the per-airport inference lock, returning 503 on timeout."""
    lock = _airport_locks[airport]
    acquired = lock.acquire(timeout=_INFERENCE_LOCK_TIMEOUT_S)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail=f"Inference engine for {airport} is busy, try again shortly.",
        )
    try:
        yield
    finally:
        lock.release()


@contextmanager
def _locked_engine(airport: str):
    """Acquire the per-airport inference lock and yield the active engine."""
    with _with_inference_lock(airport):
        yield _get_engine(airport)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint — reports per-airport status."""
    multi_manager: MultiCheckpointManager = app.state.multi_manager
    status: dict[str, str | None] = {}
    for airport in multi_manager.airports:
        status[airport] = multi_manager.get(airport).active
    missing = [ap for ap, ckpt in status.items() if ckpt is None]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Missing active checkpoint for: {', '.join(missing)}",
        )
    return {"status": "ok", "airports": status}


@app.get("/healthz")
def healthz():
    """Health check endpoint for Load Balancer."""
    return {"status": "ok"}


@app.get(
    "/checkpoints", response_model=CheckpointListResponse, dependencies=[Depends(_require_api_key)]
)
def list_checkpoints():
    """List available checkpoints per airport and the currently active ones."""
    multi_manager: MultiCheckpointManager = app.state.multi_manager
    airports_info: dict[Airport, AirportCheckpointInfo] = {}
    for airport in multi_manager.airports:
        mgr = multi_manager.get(airport)
        airports_info[Airport(airport)] = AirportCheckpointInfo(
            checkpoints=mgr.available,
            active=mgr.active,
        )
    return CheckpointListResponse(
        airports=airports_info,
        architecture=ModelConfigInfo.from_config(),
    )


@app.put("/checkpoints/{airport}/active", dependencies=[Depends(_require_api_key)])
def set_active_checkpoint(airport: Airport, request: SetCheckpointRequest):
    """Switch the active model checkpoint for a specific airport."""
    manager = app.state.multi_manager.get(airport)
    with _with_inference_lock(airport):
        try:
            manager.load(request.name)
        except ValueError as exc:
            detail = str(exc)
            status = 422 if "incompatible" in detail else 404
            raise HTTPException(status_code=status, detail=detail) from exc
        except RuntimeError as exc:
            logger.error(
                "[%s] Failed to load checkpoint %r: %s",
                airport,
                request.name,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail="Failed to load checkpoint") from exc
    return {"airport": airport, "name": manager.active, "status": "loaded"}


@app.post(
    "/checkpoints/{airport}/upload",
    response_model=UploadCheckpointResponse,
    dependencies=[Depends(_require_api_key)],
)
def upload_checkpoint(airport: Airport, file: UploadFile):
    """Upload a zipped checkpoint for a specific airport.

    The checkpoint name is derived from the zip filename (minus .zip).
    The new checkpoint is NOT set as active.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a .zip archive")

    if file.size and file.size > MAX_CHECKPOINT_ZIP_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file exceeds maximum size of {MAX_CHECKPOINT_ZIP_SIZE} bytes",
        )

    try:
        checkpoint_name = checkpoint_name_from_upload(filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    manager = app.state.multi_manager.get(airport)

    if manager.is_full():
        raise HTTPException(
            status_code=422,
            detail=(
                f"[{airport}] Checkpoint limit reached: {len(manager.available)} checkpoints "
                f"already exist. Maximum allowed is {MAX_CHECKPOINT_COUNT}."
            ),
        )

    base_dir = manager.base_dir

    final_dir = base_dir / checkpoint_name
    if final_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Checkpoint {checkpoint_name!r} already exists for {airport}",
        )

    tmp_dir = base_dir / f"_upload_tmp_{checkpoint_name}"
    try:
        extracted = tmp_dir / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        try:
            extract_checkpoint_zip(file.file, extracted)
        except (ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Determine checkpoint root: either a single top-level dir or the
        # extracted root itself.
        top_entries = list(extracted.iterdir())
        if (
            len(top_entries) == 1
            and top_entries[0].is_dir()
            and (top_entries[0] / RLLIB_MARKER).is_file()
        ):
            checkpoint_root = top_entries[0]
        elif (extracted / RLLIB_MARKER).is_file():
            checkpoint_root = extracted
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Zip does not contain a valid checkpoint ({RLLIB_MARKER} not found)",
            )

        # Validate checkpoint weight shapes against current architecture
        try:
            validate_checkpoint(checkpoint_root)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Move validated checkpoint to final location
        shutil.move(str(checkpoint_root), str(final_dir))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to process zip: {exc}") from exc
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    available = manager.discover()
    return UploadCheckpointResponse(
        airport=airport,
        name=checkpoint_name,
        checkpoints=available,
        active=manager.active,
    )


@app.delete(
    "/checkpoints/{airport}/{name}",
    response_model=CheckpointListResponse,
    dependencies=[Depends(_require_api_key)],
)
def delete_checkpoint(airport: Airport, name: str):
    """Delete a checkpoint by name for a specific airport.

    Returns 404 if the checkpoint does not exist.
    Returns 409 if the checkpoint is currently active.
    """
    manager = app.state.multi_manager.get(airport)
    if name not in manager.available:
        raise HTTPException(status_code=404, detail=f"Checkpoint {name!r} not found for {airport}")
    if name == manager.active:
        raise HTTPException(
            status_code=409,
            detail=f"Checkpoint {name!r} is currently active for {airport}; switch first",
        )
    shutil.rmtree(manager.base_dir / name)
    manager.discover()

    # Return full per-airport listing
    return list_checkpoints()


@app.post("/evaluate", response_model=InferenceResponse, dependencies=[Depends(_require_api_key)])
def evaluate(request: InferenceRequest):
    """Run multi-candidate inference and return best valid action sequence."""
    with _locked_engine(request.airport) as engine:
        candidate = run_inference_candidate(
            engine, request.world_json, "/evaluate", run_count_override=request.candidateRuns
        )
    return InferenceResponse(
        actions=[ActionRecord.model_validate(a) for a in candidate["actions"]],
        candidate={k: v for k, v in candidate.items() if k != "actions"},
    )


@app.post(
    "/suggestions",
    response_model=TaskStartResponse,
    dependencies=[Depends(_require_api_key)],
)
def suggestions(request: SuggestionsRequest):
    """Start inference as a background task. Returns a task ID for polling."""
    aircraft = request.snapshot
    logger.info(
        "[/suggestions] Received %d aircraft, snapshotTime=%ss", len(aircraft), request.snapshotTime
    )

    if len(aircraft) == 0:
        raise HTTPException(status_code=422, detail="Snapshot must contain at least one aircraft")

    aircraft_dicts = [ac.model_dump() for ac in aircraft]

    try:
        airport_splits = app.state.converter.convert_split(
            aircraft_dicts,
            snapshot_time=request.snapshotTime,
            preferred_altitude_num_levels=request.preferredAltitudeLevelsNum,
        )
    except Exception as exc:
        logger.error("Snapshot conversion failed", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Snapshot conversion error: {exc}") from exc

    if not airport_splits:
        raise HTTPException(
            status_code=422,
            detail="No convertible aircraft in snapshot (all finished or undetectable airport)",
        )

    # Validate world JSON for each airport before starting background work
    for airport_code, (world_json, _) in airport_splits.items():
        try:
            World.from_json(world_json)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid world JSON for {airport_code}: {exc}"
            ) from exc

    for airport_code, (_, callsign_map) in airport_splits.items():
        if len(callsign_map) > Config.MAX_AIRCRAFT_COUNT_INFERENCE:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Too many aircraft in snapshot: {len(callsign_map)} exceeds "
                    f"MAX_AIRCRAFT_COUNT_INFERENCE={Config.MAX_AIRCRAFT_COUNT_INFERENCE}"
                ),
            )

    run_count = (
        request.candidateRuns
        if request.candidateRuns is not None and request.candidateRuns > 0
        else int(Config.INFERENCE_CANDIDATE_RUNS)
    )
    task = _task_store.create(airports=list(airport_splits.keys()), run_count=run_count)

    # Build remaining routes per callsign for trombone fallback logic
    all_routes: dict[str, list[str]] = {}
    for ac in aircraft_dicts:
        rt = ac["routeTracker"]
        remaining = [wp["name"] for wp in rt["route"][rt["currentIndex"] :]]
        all_routes[ac["callSign"]] = remaining

    thread = threading.Thread(
        target=_run_suggestions_task,
        args=(
            task.task_id,
            airport_splits,
            all_routes,
            request.snapshotTime,
            request.candidateRuns,
            task.cancel_event,
        ),
        daemon=True,
    )
    thread.start()

    return TaskStartResponse(task_id=task.task_id)


def _run_airport_inference(
    task_id: str,
    airport_code: str,
    world_json: str,
    callsign_map: dict[str, str],
    candidate_runs_override: int | None,
    cancel_event: threading.Event,
) -> list[dict]:
    """Run inference for one airport, acquiring its per-airport lock."""

    def on_progress(completed: int, _total: int) -> None:
        _task_store.update_progress(task_id, airport_code, completed)

    with _with_inference_lock(airport_code):
        engine = _get_engine(airport_code)
        actions = run_inference(
            engine,
            world_json,
            f"/suggestions[{airport_code}]",
            run_count_override=candidate_runs_override,
            on_candidate_complete=on_progress,
            cancel_event=cancel_event,
        )

    # Remap TESTnnn callsigns back to original frontend callsigns
    reverse_map = {v: k for k, v in callsign_map.items()}
    for a in actions:
        a["call_sign"] = reverse_map.get(a["call_sign"], a["call_sign"])
    return actions


def _run_suggestions_task(
    task_id: str,
    airport_splits: dict[str, tuple[str, dict[str, str]]],
    all_routes: dict[str, list[str]],
    snapshot_time: float,
    candidate_runs_override: int | None,
    cancel_event: threading.Event,
) -> None:
    """Background worker: run inference for both airports in parallel."""
    try:
        all_actions: list[dict] = []

        with ThreadPoolExecutor(max_workers=len(airport_splits)) as executor:
            futures = {
                executor.submit(
                    _run_airport_inference,
                    task_id,
                    code,
                    wj,
                    csmap,
                    candidate_runs_override,
                    cancel_event,
                ): code
                for code, (wj, csmap) in airport_splits.items()
            }
            for future in as_completed(futures):
                all_actions.extend(future.result())

        suggestion_map, raw_actions = convert_actions_to_suggestions(
            all_actions, all_routes, snapshot_time=snapshot_time
        )
        _task_store.complete(task_id, suggestion_map, raw_actions)

    except InferenceCancelled:
        logger.info("[task %s] Cancelled", task_id)

    except NoValidSolutionError as exc:
        logger.info("[task %s] %s", task_id, exc)
        _task_store.fail(task_id, str(exc))

    except Exception as exc:
        error_msg = exc.detail if isinstance(exc, HTTPException) else str(exc)
        logger.error("[task %s] Failed: %s", task_id, error_msg, exc_info=True)
        _task_store.fail(task_id, error_msg)


@app.get(
    "/suggestions/{task_id}",
    response_model=TaskStatusResponse,
    dependencies=[Depends(_require_api_key)],
)
def get_suggestion_status(task_id: str):
    """Poll the status of a background inference task."""
    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    result_out = None
    if task.status == TaskStatus.COMPLETED and task.result is not None:
        result_out = {cs: [Suggestion(**s) for s in items] for cs, items in task.result.items()}

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress={
            code: AirportProgressInfo(
                completed_runs=prog.completed_runs,
                total_runs=prog.total_runs,
            )
            for code, prog in task.progress.items()
        },
        result=result_out,
        rawActions=task.raw_actions,
        error=task.error,
    )


@app.delete(
    "/suggestions/{task_id}",
    dependencies=[Depends(_require_api_key)],
)
def delete_suggestion_task(task_id: str):
    """Stop a running inference task and delete it."""
    deleted = _task_store.cancel_and_delete(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return {"task_id": task_id, "deleted": True}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Start the uvicorn server."""
    setup_logging()
    host = os.environ.get("INFERENCE_HOST", "0.0.0.0")
    port = int(os.environ.get("INFERENCE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    main()
