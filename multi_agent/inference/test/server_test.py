"""Unit tests for inference.server multi-run selection and multi-airport routing."""

import io
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile

import inference.server as server_module
import inference.service as service_module
from actions.actions import Action
from inference.server_models import Airport

_TEST_AIRPORT = Airport.MXP


def _candidate(
    *,
    run_index: int,
    end_reason: str | None,
    deviation: float,
    max_infringement: float,
    reward: float,
    action_count: int = 1,
    action_types: list[int] | None = None,
) -> dict[str, Any]:
    types = action_types if action_types is not None else [1] * action_count
    actions = [
        {
            "call_sign": f"TEST{run_index:03d}",
            "action": at,
            "x": 0.0,
            "y": 0.0,
            "step": run_index + 1,
            "reward": reward,
        }
        for at in types
    ]
    return {
        "run_index": run_index,
        "actions": actions,
        "end_reason": end_reason,
        "total_deviation": deviation,
        "eligible": end_reason in {"success", "delayed"} and max_infringement == 0,
        "timing_upper_bound_s": deviation,
        "command_cost": len(actions),
        "max_infringement": max_infringement,
        "cumulative_reward": reward,
    }


# ---------------------------------------------------------------------------
# Helpers — fake multi-manager with simple engine property
# ---------------------------------------------------------------------------
class _FakeCheckpointManager:
    """Fake checkpoint manager with a simple engine property."""

    def __init__(self, engine=None, active="test-ckpt"):
        self.active = active
        self._engine = engine if engine is not None else MagicMock()

    @property
    def engine(self):
        return self._engine

    def load(self, name):
        self.active = name


class _FakeMultiManager:
    def __init__(self, engine=None, active="test-ckpt"):
        self._managers: dict[str, _FakeCheckpointManager] = {}
        for airport in ["MXP", "BGY"]:
            self._managers[airport] = _FakeCheckpointManager(engine=engine, active=active)

    def get(self, airport):
        return self._managers[airport.upper()]

    @property
    def airports(self):
        return list(self._managers)


@pytest.fixture()
def _set_multi_manager(monkeypatch):
    """Install a _FakeMultiManager on app.state and restore afterwards."""
    original = getattr(server_module.app.state, "multi_manager", None)

    def _install(engine=None, active="test-ckpt"):
        fake = _FakeMultiManager(engine=engine, active=active)
        server_module.app.state.multi_manager = fake
        return fake

    yield _install

    if original is None:
        if hasattr(server_module.app.state, "multi_manager"):
            del server_module.app.state.multi_manager
    else:
        server_module.app.state.multi_manager = original


def _make_request(world_json="{}", airport=_TEST_AIRPORT, candidate_runs=None):
    return server_module.InferenceRequest(
        airport=airport, world_json=world_json, candidateRuns=candidate_runs
    )


def _setup_candidates(monkeypatch, manager_fn, candidates, **config_overrides):
    """Wire up inference mocks so evaluate() runs selection on the given candidates."""
    for key, val in config_overrides.items():
        monkeypatch.setattr(service_module.Config, key, val)
    monkeypatch.setattr(service_module.Config, "INFERENCE_CANDIDATE_RUNS", len(candidates))
    manager_fn()
    monkeypatch.setattr(
        service_module,
        "run_single_candidate",
        lambda _e, _w, _l, run_index, _s, time_offset_s=0.0: candidates[run_index],
    )


@pytest.fixture()
def _fake_world(monkeypatch):
    """Stub World.from_json so tests don't need real world JSON."""
    world = SimpleNamespace(aircraft=[], age_in_seconds=0.0, age_in_steps=0)

    class _FakeWorldParser:
        @staticmethod
        def from_json(_world_json: str):
            return world

    monkeypatch.setattr(service_module, "World", _FakeWorldParser)
    monkeypatch.setattr(service_module, "simulate_n_next_states", lambda _n, _dt, w, _c: w)
    return world


# ---------------------------------------------------------------------------
# API key tests (unchanged — don't touch engine/manager)
# ---------------------------------------------------------------------------
def test_require_api_key_allows_calls_when_auth_disabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY_ENABLED", "false")
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)

    assert server_module._require_api_key(api_key=None) is None


def test_require_api_key_rejects_missing_key_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY_ENABLED", "true")
    monkeypatch.setenv("INFERENCE_API_KEY", "secret-key")

    with pytest.raises(HTTPException, match="Invalid or missing API key") as exc_info:
        server_module._require_api_key(api_key=None)

    assert exc_info.value.status_code == 401


def test_require_api_key_rejects_invalid_key_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY_ENABLED", "true")
    monkeypatch.setenv("INFERENCE_API_KEY", "secret-key")

    with pytest.raises(HTTPException, match="Invalid or missing API key") as exc_info:
        server_module._require_api_key(api_key="wrong-key")

    assert exc_info.value.status_code == 401


def test_require_api_key_allows_valid_key_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY_ENABLED", "true")
    monkeypatch.setenv("INFERENCE_API_KEY", "secret-key")

    assert server_module._require_api_key(api_key="secret-key") is None


# ---------------------------------------------------------------------------
# Inference candidate selection tests
# ---------------------------------------------------------------------------
def test_run_inference_discards_infringement_and_prefers_success(
    monkeypatch, _set_multi_manager, _fake_world
):
    candidates = [
        _candidate(
            run_index=0,
            end_reason="success",
            deviation=80.0,
            max_infringement=0.0,
            reward=5.0,
        ),
        _candidate(
            run_index=1,
            end_reason="infringement",
            deviation=5.0,
            max_infringement=1.0,
            reward=9.0,
        ),
        _candidate(
            run_index=2,
            end_reason="delayed",
            deviation=20.0,
            max_infringement=0.0,
            reward=1.0,
        ),
    ]

    monkeypatch.setattr(service_module.Config, "INFERENCE_CANDIDATE_RUNS", len(candidates))
    _set_multi_manager()
    seen_runs: list[int] = []

    def _fake_run_single_candidate(
        _engine, _world_json, _label, run_index, _seed, time_offset_s=0.0
    ):
        seen_runs.append(run_index)
        return candidates[run_index]

    monkeypatch.setattr(service_module, "run_single_candidate", _fake_run_single_candidate)

    response = server_module.evaluate(_make_request())

    assert seen_runs == [0, 1, 2]
    # Success (run 0) beats delayed (run 2) despite higher deviation
    assert response.model_dump()["actions"] == candidates[0]["actions"]


def test_safe_success_precedes_delayed(monkeypatch, _set_multi_manager, _fake_world):
    """Safe success precedes delayed regardless of reward or command count."""
    candidates = [
        _candidate(
            run_index=0,
            end_reason="success",
            deviation=25.0,
            max_infringement=0.0,
            reward=5.0,
            action_count=5,
        ),
        _candidate(
            run_index=1,
            end_reason="delayed",
            deviation=35.0,
            max_infringement=0.0,
            reward=4.0,
            action_count=1,
        ),
    ]

    _setup_candidates(
        monkeypatch,
        _set_multi_manager,
        candidates,
    )
    response = server_module.evaluate(_make_request())
    assert response.model_dump()["actions"] == candidates[0]["actions"]


def test_conservative_timing_precedes_reward(monkeypatch, _set_multi_manager, _fake_world):
    """Conservative timing precedes reward and command cost among safe successes."""
    candidates = [
        _candidate(
            run_index=0,
            end_reason="success",
            deviation=10.0,
            max_infringement=0.0,
            reward=14.0,
            action_count=7,
        ),
        _candidate(
            run_index=1,
            end_reason="success",
            deviation=11.0,
            max_infringement=0.0,
            reward=8.0,
            action_count=1,
        ),
    ]

    _setup_candidates(
        monkeypatch,
        _set_multi_manager,
        candidates,
    )
    response = server_module.evaluate(_make_request())
    assert response.model_dump()["actions"] == candidates[0]["actions"]


def test_trombone_commands_count_in_tie_break(monkeypatch, _set_multi_manager, _fake_world):
    """All learned commands, including trombones, count in the cost tie break."""
    candidates = [
        _candidate(
            run_index=0,
            end_reason="success",
            deviation=10.0,
            max_infringement=0.0,
            reward=6.0,
            action_types=[
                int(Action.SLOW_DOWN_SMALL),
                int(Action.LENGTHEN_TROMBONE),
                int(Action.LENGTHEN_TROMBONE),
                int(Action.LENGTHEN_TROMBONE),
                int(Action.SHORTEN_TROMBONE),
            ],
        ),
        _candidate(
            run_index=1,
            end_reason="success",
            deviation=10.0,
            max_infringement=0.0,
            reward=6.0,
            action_types=[
                int(Action.SLOW_DOWN_SMALL),
                int(Action.SLOW_DOWN_SMALL),
                int(Action.SLOW_DOWN_SMALL),
            ],
        ),
    ]

    _setup_candidates(
        monkeypatch,
        _set_multi_manager,
        candidates,
    )
    response = server_module.evaluate(_make_request())
    assert response.model_dump()["actions"] == candidates[1]["actions"]


def test_exact_score_ties_use_deterministic_run_index(monkeypatch, _set_multi_manager, _fake_world):
    """Equal success, timing and command cost use stable run order."""
    candidates = [
        _candidate(
            run_index=0,
            end_reason="success",
            deviation=10.0,
            max_infringement=0.0,
            reward=4.0,
            action_types=[int(Action.SLOW_DOWN_LARGE)],
        ),
        _candidate(
            run_index=1,
            end_reason="success",
            deviation=10.0,
            max_infringement=0.0,
            reward=4.0,
            action_types=[int(Action.SLOW_DOWN_SMALL)],
        ),
    ]

    _setup_candidates(
        monkeypatch,
        _set_multi_manager,
        candidates,
    )
    response = server_module.evaluate(_make_request())
    assert response.model_dump()["actions"] == candidates[0]["actions"]


def test_run_inference_discards_timeout_candidates(monkeypatch, _set_multi_manager, _fake_world):
    candidates = [
        _candidate(
            run_index=0,
            end_reason="timeout",
            deviation=5.0,
            max_infringement=0.0,
            reward=10.0,
        ),
        _candidate(
            run_index=1,
            end_reason="success",
            deviation=50.0,
            max_infringement=0.0,
            reward=2.0,
        ),
    ]

    _setup_candidates(monkeypatch, _set_multi_manager, candidates)
    response = server_module.evaluate(_make_request())
    # Timeout (run 0) is discarded despite better deviation
    assert response.model_dump()["actions"] == candidates[1]["actions"]


def test_run_inference_raises_if_no_valid_candidates(monkeypatch, _set_multi_manager, _fake_world):
    candidates = [
        _candidate(
            run_index=0,
            end_reason="infringement",
            deviation=30.0,
            max_infringement=1.0,
            reward=-1.0,
        ),
        _candidate(
            run_index=1,
            end_reason="timeout",
            deviation=25.0,
            max_infringement=0.0,
            reward=-2.0,
        ),
    ]

    _setup_candidates(monkeypatch, _set_multi_manager, candidates)
    with pytest.raises(
        service_module.NoValidSolutionError, match="No valid inference solution found"
    ):
        server_module.evaluate(_make_request())


def test_run_inference_uses_random_seed_per_candidate(monkeypatch, _set_multi_manager, _fake_world):
    run_count = 3
    seeds = [101, 202, 303]
    monkeypatch.setattr(service_module.Config, "INFERENCE_CANDIDATE_RUNS", run_count)
    _set_multi_manager()
    monkeypatch.setattr(service_module, "draw_candidate_seeds", lambda count: seeds[:count])

    received_seeds: list[int] = []

    def _fake_run_single_candidate(
        _engine, _world_json, _label, run_index, seed, time_offset_s=0.0
    ):
        received_seeds.append(seed)
        return _candidate(
            run_index=run_index,
            end_reason="success",
            deviation=float(run_index + 1),
            max_infringement=0.0,
            reward=1.0,
        )

    monkeypatch.setattr(service_module, "run_single_candidate", _fake_run_single_candidate)

    server_module.evaluate(_make_request())

    assert received_seeds == seeds


def test_candidates_receive_correct_temporal_offsets(monkeypatch, _set_multi_manager, _fake_world):
    """Each candidate's world is advanced by the expected offset from the configured span."""
    run_count = 3
    span_s = 60
    monkeypatch.setattr(service_module.Config, "INFERENCE_CANDIDATE_RUNS", run_count)
    monkeypatch.setattr(service_module.Config, "INFERENCE_CANDIDATE_SPAN_S", span_s)
    _set_multi_manager()

    received_offsets: list[int] = []
    monkeypatch.setattr(
        service_module,
        "simulate_n_next_states",
        lambda n, _dt, w, _c: (received_offsets.append(n), w)[1],
    )

    monkeypatch.setattr(
        service_module,
        "run_single_candidate",
        lambda _e, _w, _l, ri, _s, time_offset_s=0.0: _candidate(
            run_index=ri, end_reason="success", deviation=1.0, max_infringement=0.0, reward=1.0
        ),
    )

    server_module.evaluate(_make_request())

    # offset_s = run_index * 60 / 3 → [0, 20, 40]; index 0 skipped (offset=0)
    assert received_offsets == [20, 40]


# ---------------------------------------------------------------------------
# Concurrency / locking tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=False)
def _fresh_lock():
    """Ensure the per-airport inference locks are released before and after each lock test."""
    for lock in server_module._airport_locks.values():
        try:
            lock.release()
        except RuntimeError:
            pass
    yield
    for lock in server_module._airport_locks.values():
        try:
            lock.release()
        except RuntimeError:
            pass


def test_concurrent_evaluate_calls_are_serialized(
    monkeypatch, _set_multi_manager, _fresh_lock, _fake_world
):
    """Two concurrent /evaluate calls must execute serially (not overlap)."""
    execution_log: list[tuple[str, float]] = []
    ready = threading.Event()

    def _slow_run_single_candidate(_engine, _wj, _label, run_index, _seed, time_offset_s=0.0):
        execution_log.append(("start", time.monotonic()))
        ready.set()  # signal that this call has started
        time.sleep(0.15)
        execution_log.append(("end", time.monotonic()))
        return _candidate(
            run_index=run_index,
            end_reason="success",
            deviation=1.0,
            max_infringement=0.0,
            reward=1.0,
        )

    monkeypatch.setattr(service_module.Config, "INFERENCE_CANDIDATE_RUNS", 1)
    _set_multi_manager()
    monkeypatch.setattr(service_module, "run_single_candidate", _slow_run_single_candidate)

    results: list[Any] = [None, None]
    errors: list[Exception | None] = [None, None]

    def _call(idx):
        try:
            results[idx] = server_module.evaluate(_make_request())
        except Exception as exc:
            errors[idx] = exc

    t1 = threading.Thread(target=_call, args=(0,))
    t2 = threading.Thread(target=_call, args=(1,))
    t1.start()
    # Wait for the first call to actually start inside run_single_candidate,
    # then launch the second so it blocks on the lock.
    ready.wait(timeout=5)
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert errors == [None, None], f"Unexpected errors: {errors}"

    # With serialization, the second call's start must be >= the first call's end.
    starts = [t for label, t in execution_log if label == "start"]
    ends = [t for label, t in execution_log if label == "end"]
    assert len(starts) == 2
    assert len(ends) == 2
    # The later start must be after the earlier end (serialized, not overlapping).
    assert max(starts) >= min(ends), f"Calls overlapped: starts={starts}, ends={ends}"


def test_lock_timeout_returns_503(monkeypatch, _set_multi_manager, _fresh_lock):
    """When the lock cannot be acquired within the timeout, return 503."""
    monkeypatch.setattr(server_module, "_INFERENCE_LOCK_TIMEOUT_S", 0.01)
    _set_multi_manager()

    # Hold the per-airport lock so the endpoint can't acquire it.
    server_module._airport_locks[_TEST_AIRPORT.value].acquire()

    with pytest.raises(HTTPException) as exc_info:
        server_module.evaluate(_make_request())

    assert exc_info.value.status_code == 503
    assert "busy" in exc_info.value.detail.lower()

    server_module._airport_locks[_TEST_AIRPORT.value].release()


def test_checkpoint_switch_waits_for_inference(
    monkeypatch, _set_multi_manager, _fresh_lock, _fake_world
):
    """PUT /checkpoints/active must wait for a running inference to finish."""
    inference_started = threading.Event()
    inference_proceed = threading.Event()
    checkpoint_loaded = threading.Event()

    def _blocking_run_single_candidate(_engine, _wj, _label, run_index, _seed, time_offset_s=0.0):
        inference_started.set()
        inference_proceed.wait(timeout=5)
        return _candidate(
            run_index=run_index,
            end_reason="success",
            deviation=1.0,
            max_infringement=0.0,
            reward=1.0,
        )

    monkeypatch.setattr(service_module.Config, "INFERENCE_CANDIDATE_RUNS", 1)
    monkeypatch.setattr(service_module, "run_single_candidate", _blocking_run_single_candidate)

    fake = _set_multi_manager(active="ckpt-a")
    mgr = fake.get(_TEST_AIRPORT.value)
    original_load = mgr.load

    def _tracking_load(name):
        original_load(name)
        checkpoint_loaded.set()

    mgr.load = _tracking_load

    # Start an inference call that blocks
    inference_result: list[Any] = [None]

    def _run_inference():
        try:
            inference_result[0] = server_module.evaluate(_make_request())
        except Exception as exc:
            inference_result[0] = exc

    t_inference = threading.Thread(target=_run_inference)
    t_inference.start()
    inference_started.wait(timeout=5)

    # Now try to switch checkpoint — it should block because inference holds the lock
    checkpoint_result: list[Any] = [None]

    def _switch_checkpoint():
        try:
            checkpoint_result[0] = server_module.set_active_checkpoint(
                airport=_TEST_AIRPORT,
                request=server_module.SetCheckpointRequest(name="ckpt-b"),
            )
        except Exception as exc:
            checkpoint_result[0] = exc

    t_checkpoint = threading.Thread(target=_switch_checkpoint)
    t_checkpoint.start()

    # Give the checkpoint thread a moment to block on the lock
    time.sleep(0.1)
    assert not checkpoint_loaded.is_set(), "Checkpoint loaded while inference was running"

    # Release inference
    inference_proceed.set()
    t_inference.join(timeout=5)
    t_checkpoint.join(timeout=5)

    assert checkpoint_loaded.is_set(), "Checkpoint was never loaded"
    assert mgr.active == "ckpt-b"


# ---------------------------------------------------------------------------
# Background task / polling tests
# ---------------------------------------------------------------------------
def test_get_suggestion_status_returns_404_for_unknown_task():
    """GET /suggestions/{task_id} returns 404 when the task ID doesn't exist."""
    with pytest.raises(HTTPException) as exc_info:
        server_module.get_suggestion_status("nonexistent")

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_delete_suggestion_task_returns_404_for_unknown_task():
    """DELETE /suggestions/{task_id} returns 404 when the task ID doesn't exist."""
    with pytest.raises(HTTPException) as exc_info:
        server_module.delete_suggestion_task("nonexistent")

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_delete_suggestion_task_cancels_and_removes_task():
    """DELETE /suggestions/{task_id} sets cancel event and removes the task."""
    task = server_module._task_store.create(airports=["MXP"], run_count=3)
    cancel_event = task.cancel_event

    result = server_module.delete_suggestion_task(task.task_id)

    assert result == {"task_id": task.task_id, "deleted": True}
    assert cancel_event.is_set()
    assert server_module._task_store.get(task.task_id) is None


def test_upload_rejects_path_traversal_name(_set_multi_manager):
    _set_multi_manager()
    upload = UploadFile(io.BytesIO(b""), filename="../evil.zip")
    with pytest.raises(HTTPException) as excinfo:
        server_module.upload_checkpoint(_TEST_AIRPORT, upload)
    assert excinfo.value.status_code == 422


def test_upload_rejects_corrupt_zip_as_unprocessable(_set_multi_manager, tmp_path):
    manager = _set_multi_manager().get(_TEST_AIRPORT)
    manager.is_full = lambda: False
    manager.base_dir = tmp_path
    upload = UploadFile(io.BytesIO(b"not a zip archive"), filename="corrupt.zip")
    with pytest.raises(HTTPException) as excinfo:
        server_module.upload_checkpoint(_TEST_AIRPORT, upload)
    assert excinfo.value.status_code == 422
    assert not any(tmp_path.iterdir())
