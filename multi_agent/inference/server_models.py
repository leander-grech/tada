"""Pydantic models and type definitions for the inference server."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from config.config import Config
from inference.task_store import TaskStatus


# ---------------------------------------------------------------------------
# Airport enum
# ---------------------------------------------------------------------------
class Airport(StrEnum):
    MXP = "MXP"
    BGY = "BGY"


BGY_PREFERRED_ALTITUDE_NUM_LEVELS = 4


# ---------------------------------------------------------------------------
# Pydantic models – /evaluate
# ---------------------------------------------------------------------------
class InferenceRequest(BaseModel):
    """Accepts a raw JSON string representing a World object."""

    airport: Airport
    world_json: str
    candidateRuns: int | None = None


class ActionRecord(BaseModel):
    """Single non-zero action returned by the model."""

    call_sign: str
    action: int
    x: float
    y: float
    step: int
    reward: float


class InferenceResponse(BaseModel):
    """List of non-zero actions produced during the episode."""

    actions: list[ActionRecord]
    candidate: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Pydantic models – /suggestions response
# ---------------------------------------------------------------------------
class SuggestionTimeToExecute(BaseModel):
    "Optional timing for when to execute a suggestion"

    start: int
    duration: int


class Suggestion(BaseModel):
    "A single suggestion for the frontend, converted from a model action."

    eventType: str
    eventArgs: list
    timeToExecute: SuggestionTimeToExecute | None = None
    rawAction: dict | None = None


# ---------------------------------------------------------------------------
# Pydantic models – /suggestions (frontend SerializedAircraft types)
# ---------------------------------------------------------------------------
class SnapshotAircraftState(BaseModel):
    """Aircraft state payload from frontend snapshot."""

    heading: float
    speed: float
    altitude: float
    turnDirection: int
    position: list[float]
    status: int
    controlMode: str
    controlStatus: int
    cautionState: str
    waypointSpeedLimit: float | None = None


class SnapshotTargetState(BaseModel):
    """Target state payload from frontend snapshot."""

    heading: float
    speed: float
    altitude: float


class SnapshotWaypoint(BaseModel):
    """Waypoint payload in frontend route tracker."""

    name: str
    location: list[float]
    min_altitude: float | None = None
    max_altitude: float | None = None
    speed: float | None = None


class SnapshotRouteTracker(BaseModel):
    """Route tracker payload from frontend snapshot."""

    currentIndex: int
    route: list[SnapshotWaypoint]


class SerializedAircraft(BaseModel):
    """Frontend serialized aircraft payload for /suggestions."""

    callSign: str
    secondsToStart: float
    manuallyUpdatedSpeed: bool
    identified: bool
    state: SnapshotAircraftState
    targetState: SnapshotTargetState
    routeTracker: SnapshotRouteTracker
    scheduledCommands: list = Field(default_factory=list)
    tadaSuggestions: Any | None = (
        None  # Can be any, if aircraft has tada suggestions we overwrite them
    )
    amanArrivalTime: float
    amanOffsetSeconds: float
    historyPath: list[list[float]] | None = None
    preferredAltitudeLevel: int | None = None


class SuggestionsRequest(BaseModel):
    """Wrapper for /suggestions endpoint accepting snapshot + timing info."""

    snapshot: list[SerializedAircraft]
    snapshotTime: float  # Frontend's current scenario time in seconds
    candidateRuns: int | None = None
    preferredAltitudeLevelsNum: int = Field(default=BGY_PREFERRED_ALTITUDE_NUM_LEVELS, gt=0)


# ---------------------------------------------------------------------------
# Pydantic models – /checkpoints
# ---------------------------------------------------------------------------
class ModelConfigInfo(BaseModel):
    """Architecture dimensions the server expects from checkpoints."""

    gru_hidden_size: int
    combined_encoder_out: int
    mlp_hidden: int
    num_actions: int
    aircraft_scalar_out: int
    flight_plan_output: int  # FLIGHT_PLAN_HIDDEN // 2
    nei_d_model: int

    @classmethod
    def from_config(cls) -> "ModelConfigInfo":
        """Construct from current Config constants."""
        return cls(
            gru_hidden_size=Config.GRU_HIDDEN_SIZE,
            combined_encoder_out=Config.COMBINED_ENCODER_OUT,
            mlp_hidden=Config.MLP_HIDDEN,
            num_actions=Config.NUM_ACTIONS,
            aircraft_scalar_out=Config.AIRCRAFT_SCALAR_OUT,
            flight_plan_output=Config.FLIGHT_PLAN_HIDDEN // 2,
            nei_d_model=Config.NEI_D_MODEL,
        )


class AirportCheckpointInfo(BaseModel):
    """Per-airport checkpoint availability and active model."""

    checkpoints: list[str]
    active: str | None


class CheckpointListResponse(BaseModel):
    """Response for GET /checkpoints."""

    airports: dict[Airport, AirportCheckpointInfo]
    architecture: ModelConfigInfo


class SetCheckpointRequest(BaseModel):
    """Request body for PUT /checkpoints/active."""

    name: str


class UploadCheckpointResponse(BaseModel):
    """Response for POST /checkpoints/upload."""

    airport: Airport
    name: str
    checkpoints: list[str]
    active: str | None


# ---------------------------------------------------------------------------
# Pydantic models – /suggestions background tasks
# ---------------------------------------------------------------------------
class TaskStartResponse(BaseModel):
    """Returned by POST /suggestions when starting a background inference task."""

    task_id: str


class AirportProgressInfo(BaseModel):
    """Per-airport inference progress."""

    completed_runs: int
    total_runs: int


class TaskStatusResponse(BaseModel):
    """Returned by GET /suggestions/{task_id}."""

    task_id: str
    status: TaskStatus
    progress: dict[str, AirportProgressInfo]
    result: dict[str, list[Suggestion]] | None = None
    rawActions: dict[str, list[dict]] | None = None
    error: str | None = None
