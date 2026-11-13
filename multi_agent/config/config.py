# pylint: disable=too-many-locals

"""Module for loading the configuration"""

import os
from datetime import date

from dotenv import load_dotenv
from flight_simulator import MissionDifficulty

from config.observation_fields import AIRCRAFT_SCALAR_FIELDS, GLOBAL_CONTEXT_FIELDS

load_dotenv()


# pylint: disable-next=too-few-public-methods
class Config:
    """Configuration class for the environment"""

    # ------------- Reproducibility -------------
    SEED: int = 42  # Global seed for deterministic training.

    # ------------- Experiment configuration -------------
    # Date-based default run name to avoid generic placeholders (e.g. "default_*").
    # Pinned into the env on first import so Ray child processes spawned after
    # midnight still resolve the same logging experiment name as the parent.
    EXPERIMENT_NAME = os.environ.setdefault("TADA_EXPERIMENT_NAME", date.today().isoformat())
    LOGGING_BACKEND = "mlflow"  # Allowed values: "mlflow", "tensorboard"
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME = None
    MLFLOW_SAVE_ARTIFACT = True
    MLFLOW_LOG_PARAMS_ON_TRIAL_END = False
    MLFLOW_TRACKING_TOKEN = None

    # ------------- Policy configuration -------------
    POLICY_NAME = "learning_policy"  # The name of the policy to train

    # ------------- Runtime configuration -------------
    EVAL_ENV_RUNNERS = 8  # Historical offline evaluation worker cap
    CUSTOM_METRICS_WINDOW = 20  # Rolling window for callback mean/min/max summary metrics
    INFERENCE_CANDIDATE_RUNS = 5  # Number of server inference candidates per request
    INFERENCE_CANDIDATE_SPAN_S = (
        60  # Time span (seconds) over which inference candidates are spread
    )

    # ------------- Training configuration -------------
    # Typical scenarios complete in 30-50 steps; 128 is a safe upper bound that
    # prevents runaway episodes while giving the agent room on harder scenarios.
    EPISODE_STEPS = 128  # Maximum steps per episode
    TRAIN_BATCH_SIZE = 16384  # Training batch size per learner (train_batch_size_per_learner)
    TRAIN_MINIBATCH_SIZE = 2048  # Mini-batch size for SGD updates (8 minibatches/epoch)

    # ------------- Environment configuration -------------
    MAX_AIRCRAFT_COUNT_EVAL = 20  # The maximum number of aircraft used during evaluation
    MAX_AIRCRAFT_COUNT_INFERENCE = 20  # The maximum number of aircraft in inference (server)
    MAX_WAYPOINTS = 20  # The maximum number of waypoints in flight plan for single aircraft
    # Aircraft scalar feature counts
    # Base features: vel_xy, vz, heading_sin/cos, time_to_target,
    # deviation_proxy, under_control, time_since_last_action
    AIRCRAFT_SCALAR_LEN = len(AIRCRAFT_SCALAR_FIELDS)
    GLOBAL_CONTEXT_DIM = len(GLOBAL_CONTEXT_FIELDS)  # Shared critic-side global context features
    FLIGHT_PLAN_WAYPOINT_DIM = 8  # Number of features in flight plan waypoint vector
    SATISFACTORY_TIME_DEV = 30.0  # Time deviation threshold for satisfactory completion
    OPERATIONAL_SUCCESS_S = 60.0  # Evaluation success window; SATISFACTORY_TIME_DEV shapes reward
    LOCKOUT_DEVIATION_THRESHOLD_S = 30.0  # Lock out aircraft with |deviation| < this from selection
    AUTO_SOLVE_TARGET_WINDOW_S = 60.0  # Stop escalating once arrival is within this early window

    # ------------- Simulator configuration -------------
    TIME_BETWEEN_ACTIONS = 45  # Time between actions in seconds
    SIMULATOR_CONFIG_ROOT = "./config/simulation_config/"
    ORIGIN_LAT = 60.0  # Origin latitude for scenario recording
    ORIGIN_LON = 25.0  # Origin longitude for scenario recording

    # ------------- Scenario generation configuration -------------
    WORLD_JSON_FILE = None  # Path to the world file to override scenario generation
    DEFAULT_TRAINING_DIFFICULTY = MissionDifficulty.VALIDATION_USE_CASE_2  # Difficulty for train
    DEFAULT_EVALUATION_DIFFICULTY = MissionDifficulty.VALIDATION_USE_CASE_2  # Difficulty for eval
    SCENARIO_EVAL_SEEDS = [
        41,
        786551703,
        1595180635,
        1001495968,
        1123251507,
        921959045,
    ]  # Seeds for evaluation. In form of [seed1, seed2, ...]
    SCENARIO_START_TIME_OFFSET_MAX_S = (
        45  # Randomized scenario start offset range [0, max] (one step)
    )

    MAX_AIRCRAFT_COUNT_TRAINING = 12  # Unbound observation capacity

    # ------------- Neighbor observation/encoder configuration -------------
    MAX_NEIGHBORS = 5  # Number of neighbors to include (fixed K)
    NEIGHBOR_LOOKAHEAD_MULTIPLIER = 6  # How many TIME_BETWEEN_ACTIONS to look ahead
    NEIGHBOR_INFRINGEMENT_WINDOW_S = (
        NEIGHBOR_LOOKAHEAD_MULTIPLIER * TIME_BETWEEN_ACTIONS
    )  # Keep neighbor severity horizon aligned with kinematic lookahead

    # ------------- Conflict detection configuration -------------
    HORIZONTAL_SAFEZONE_RADIUS_NM = 5.0  # Horizontal safezone radius in nautical miles
    HORIZONTAL_CONFLICT_RADIUS_NM = 3.0  # Horizontal conflict radius in nautical miles
    VERTICAL_SAFEZONE_RADIUS_FT = 1000.0  # Vertical safezone radius in feet
    VERTICAL_CONFLICT_RADIUS_FT = 500.0  # Vertical conflict radius in feet
    NEAR_CONFLICT_THRESHOLD = 0.7  # Severity threshold for metrics, observations, and lockout
    REWARD_INFRINGEMENT_GATE_CENTER = 0.5  # Sigmoid gate center for infringement reward shaping

    # ------------- Action configuration -------------
    NUM_ACTIONS = 15  # The number of actions available to the agent (actions/actions.py)
    ACTION_HEADING_ADJUSTMENT = 40.0  # The maximum heading adjustment in degrees during vectoring
    SPEED_RESOLUTION = 10.0  # Speed commands are rounded to this granularity (knots)
    SPEED_LIMIT_BUFFER_KT = 10.0  # Mask buffer from aircraft design min/max speed limits
    ACTION_SPEED_DELTA_SMALL = 10.0  # The maximum speed adjustment in knots
    ACTION_SPEED_DELTA_MEDIUM = 20.0  # The maximum speed adjustment in knots
    ACTION_SPEED_DELTA_LARGE = 40.0  # The maximum speed adjustment in knots
    WAYPOINTS_TO_LEAVE_AT_MINIMUM = 4  # Minimum waypoints to keep when skipping
    MAX_TURN_ANGLE_AFTER_HEADING_ADJUSTMENT = 90.0  # Turn angle to return to route after vectoring
    SKIP_MAX_INTERCEPT_ANGLE_DEG = 95.0  # Max angle to intercept remaining route after skip
    SKIP_MIN_DEVIATION_DEG = 15.0  # Min route turn at skipped waypoint for skip to be meaningful
    TROMBONE_EXCLUSION_SHRINK_FACTOR = (
        0.8  # Shrink trombone hull to centre corridor (0=point, 1=full)
    )

    # Optional per-airport custom action mask, 1 = allowed, 0 = blocked. Airport code → list of length Config.NUM_ACTIONS or None.
    # Missing airports (and None values) fall through to the natively computed mask (no override).
    # DO_NOTHING (index 0) must stay on where used — the compute_action_mask assertion forbids empty masks.
    CUSTOM_ACTION_MASK_BY_AIRPORT: dict[str, list[int] | None] = {
        "MXP": [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "BGY": [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    }

    # ------------- Aircraft selection configuration -------------
    ACTION_SELECTION_MODE = "sector_based"  # "single": one global actor/step, "sector_based": up to one actor per configured sector group/step
    SECTOR_SELECTION_GROUPS: list[list[str]] = [
        ["ANE", "ANW1", "ANW2"],
        ["ASW", "MAP", "MAR"],
        ["ADE1", "ADE2", "ADE3", "LAR"],
    ]  # Later groups win when sector polygons overlap.
    MAX_ACTING_AGENTS_BASE = 1  # Used only when ACTION_SELECTION_MODE == "single"
    SELECTION_WEIGHT_CONFLICT = 3.0  # Weight for conflict severity in agent selection
    SELECTION_WEIGHT_DEVIATION = 1.0  # Weight for time deviation in agent selection
    SELECTION_COOLDOWN_STEPS = 1  # Stage-C fairness cooldown after an agent is selected
    SELECTION_RECENT_NOOP_HOLDOFF_STEPS = (
        1  # Steps to suppress reselection after an explicit do-nothing (had choices but passed).
        # Raise to reduce wasted cycles on idle aircraft; lower/0 to let the model reconsider sooner.
    )
    SELECTION_SOFT_SUPPRESSION_DOMINANCE_MARGIN = (
        0.5  # Score gap required for a suppressed agent to bypass cooldown or noop holdoff.
        # Raise to make suppression stickier; lower toward 0 to let urgency override more easily.
    )
    MAX_CONSECUTIVE_ACTS = 2  # Maximum consecutive actions per agent (more bandwidth at high AC)

    # ------------- Normalization configuration -------------
    # Position normalization (in nautical miles and feet)
    REL_POS_XY_MAX = 100.0  # in nautical miles (same for X and Y)

    RENDER_POS_XY_MIN, RENDER_POS_XY_MAX = -100.0, 100.0  # for rendering only

    # Relative altitude normalization window (+/- value in feet)
    # Covers cruise-to-ground altitude differences (e.g. FL350 waypoint to sea-level arrival)
    REL_POS_Z_RANGE_FT = 40000.0

    # Velocity normalization (in knots and feet per minute)
    VEL_XY_MAX = 350.0  # knots (same for X and Y horizontal velocity)
    VEL_Z_MAX = 2000.0  # feet per minute (vertical velocity)

    # Relative feature normalization ranges
    REL_VEL_RANGE_KTS = 500.0
    REL_VZ_RANGE_FPM = 1000.0  # vertical speed difference range (feet per minute)
    TTCA_MAX_S = NEIGHBOR_LOOKAHEAD_MULTIPLIER * TIME_BETWEEN_ACTIONS  # Time to collision ahead
    DCPA_MAX_NM = (
        NEIGHBOR_LOOKAHEAD_MULTIPLIER * TIME_BETWEEN_ACTIONS * VEL_XY_MAX / 3600.0
    )  # Distance to collision ahead

    # Time normalization (in seconds/minutes)
    TIME_TO_TARGET_MAX = (
        EPISODE_STEPS * TIME_BETWEEN_ACTIONS
    )  # Match training decision horizon for better dynamic range
    TIME_DEVIATION_MAX = 900.0  # max deviation (TTG/TTG)
    DEVIATION_ENCODING = "linear"  # Select signed_log only for a versioned fresh run.
    DEVIATION_KNEE_S = 60.0
    DEVIATION_CAP_S = 900.0
    NEIGHBOR_SELECTION = "risk"
    TIME_SINCE_LAST_ACTION_MAX = (
        EPISODE_STEPS * TIME_BETWEEN_ACTIONS
    )  # max elapsed time for time_since_last_action feature

    # ------------- Model architecture configuration -------------
    MLP_HIDDEN = 96  # Hidden layer size for policy and value heads

    # Aircraft scalar encoder (single MLP: scalar_len → hidden → out)
    AIRCRAFT_SCALAR_HIDDEN = 160  # Hidden layer size for aircraft scalar encoder
    AIRCRAFT_SCALAR_OUT = 96  # Output size for aircraft scalar encoder

    # Flight plan encoder (1D CNN)
    FLIGHT_PLAN_HIDDEN = 192  # Hidden layer size for flight plan encoder
    FLIGHT_PLAN_CONV1_CHANNELS = 48  # Number of channels for first conv layer
    FLIGHT_PLAN_CONV2_CHANNELS = 96  # Number of channels for second conv layer

    # Neighbor encoder
    NEIGHBOR_FEATURE_DIM = 16  # Dn: number of per-neighbor features (14 kinematic + 1 infringement + 1 head-on collision)
    NEI_D_MODEL = 128  # Neighbor encoder hidden size
    NEI_ATTN_HEADS = 4  # Number of attention heads for neighbor encoder (must divide NEI_D_MODEL)

    # Combined encoder
    COMBINED_ENCODER_HIDDEN = 384  # Hidden layer size for combined encoder
    COMBINED_ENCODER_OUT = 192  # Output size for combined encoder

    # GRU temporal memory
    GRU_HIDDEN_SIZE = 192  # Hidden size for GRU layer
    GRU_MAX_SEQ_LEN = 8  # Max sequence length for recurrent training (BPTT window)
    CRITIC_CONTEXT_OUT = 64  # Critic-only projection width for global context

    # ------------- Reward configuration -------------
    # Maximum points awarded for fully mitigating the delay to zero (per episode upper bound)
    # The reward function grants incremental progress so that reaching zero delay yields at most
    # this many points in total (before action penalties).
    REWARD_MAX_FOR_ZERO_DELAY = 10.0  # Reward for fully mitigating the delay to zero
    REWARD_MAX_FOR_ZERO_INFRINGEMENT = 15.0  # Reward for fully mitigating the infringement to zero
    REWARD_ACTION_PENALTY_WEIGHT = 1.0  # Weight for action penalty in reward calculation
    ACTION_DAMPING_THRESHOLD_STEPS = (
        3
        * GRU_MAX_SEQ_LEN  # Steps before action-frequency penalty fully decays (Carvell et al. 2026)
    )
    SPEED_DAMPING_THRESHOLD_STEPS = (
        5 * GRU_MAX_SEQ_LEN  # Steps before speed-frequency penalty fully decays
    )
    REWARD_SPEED_PENALTY_WEIGHT = 1.0  # Weight for speed-specific damping penalty
    REWARD_GLOBAL_SHARE = (
        0.08  # Fraction of step reward shared with all active agents for cooperation
    )
    SUCCESS_REWARD_BONUS = 3.0  # Bonus reward for successful completion
    SUCCESS_MARGIN_BONUS_MAX = 1.0  # Extra terminal bonus for exceeding success thresholds
    DELAYED_CLEAN_BONUS = 1.0  # Reward for conflict-free delayed episodes (delay is OK if safe)


def get_logging_backend() -> str:
    """Return normalized logging backend with validation."""
    backend = str(Config.LOGGING_BACKEND).strip().lower()
    allowed = {"mlflow", "tensorboard"}
    if backend not in allowed:
        raise ValueError(
            f"Invalid LOGGING_BACKEND={Config.LOGGING_BACKEND!r}. "
            "Allowed values: 'mlflow', 'tensorboard'."
        )
    return backend


assert Config.EPISODE_STEPS / Config.GRU_MAX_SEQ_LEN <= 16.0, (
    "GRU_MAX_SEQ_LEN is too small relative to EPISODE_STEPS. "
    "Keep EPISODE_STEPS / GRU_MAX_SEQ_LEN <= 16.0 to limit recurrent truncation mismatch."
)

assert Config.TRAIN_BATCH_SIZE / Config.GRU_MAX_SEQ_LEN >= 512.0, (
    "GRU_MAX_SEQ_LEN is too large for the current TRAIN_BATCH_SIZE. "
    "Keep at least ~512 recurrent sequences per learner batch."
)

assert Config.TRAIN_MINIBATCH_SIZE / Config.GRU_MAX_SEQ_LEN >= 64.0, (
    "GRU_MAX_SEQ_LEN is too large for the current TRAIN_MINIBATCH_SIZE. "
    "Keep at least ~64 recurrent sequences per minibatch."
)
