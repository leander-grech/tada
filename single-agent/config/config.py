# pylint: disable=too-many-locals

"""Module for loading the configuration"""


import os
from pathlib import Path

from dotenv import load_dotenv
from flight_simulator import MissionDifficulty

load_dotenv()


# Default evaluation seeds, used when no eval-seeds file is present (see _load_eval_seeds).
_DEFAULT_EVAL_SEEDS = [
    41,
    786551703,
    1595180635,
    1001495968,
    1123251507,
]

# Optional external eval-seed pool, resolved relative to THIS file so it is cwd-independent.
# Drop integers into config/eval_seeds.txt (one per line; blank lines and `#...` comments allowed;
# commas / whitespace / JSON-array brackets all tolerated) to evaluate on a larger set without
# editing config.py. Missing / seedless / malformed file => the defaults above are used.
EVAL_SEEDS_FILE = Path(__file__).parent / "eval_seeds.txt"


def _load_eval_seeds(path: Path, default: list[int]) -> list[int]:
    """Return seeds parsed from `path`, or `default` if it is missing / empty / unparseable.

    Tolerant format: integers separated by newlines, commas, or whitespace; `#` starts a line
    comment; surrounding `[ ]` brackets are ignored (so a pasted Python/JSON list also parses).
    Order is preserved (eval walks the pool in order); duplicates are kept as-is.
    """
    if not path.exists():
        return list(default)
    try:
        seeds: list[int] = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.split("#", 1)[0]  # drop trailing comments
            for tok in line.replace("[", " ").replace("]", " ").replace(",", " ").split():
                seeds.append(int(tok))
    except (ValueError, OSError) as err:  # malformed / unreadable: fall back, don't crash startup
        print(f"[config] WARNING: could not parse {path} ({err}); using {len(default)} default eval seeds")
        return list(default)
    if not seeds:
        return list(default)
    print(f"[config] loaded {len(seeds)} eval seeds from {path}")
    return seeds


# pylint: disable-next=too-few-public-methods
class Config:
    """Configuration class for the environment"""

    # ------------- Hardware configuration -------------

            # ------------- Training configuration -------------
    EPISODE_STEPS = 64  # Fixed-horizon fallback / cap reference (see per-scenario horizon below)
    # --- Per-scenario episode horizon ---
    # Most timeouts are structural: 4/5 eval scenarios have a latest arrival ETA beyond the
    # fixed 64*45s=2880s horizon, so aircraft physically cannot all land in time even under a
    # do-nothing policy. Size each episode to the scenario's latest arrival ETA instead:
    #   max_steps = clip(ceil(max_arrival_eta / TIME_BETWEEN_ACTIONS) + HORIZON_STEP_MARGIN,
    #                    MIN_EPISODE_STEPS, MAX_EPISODE_STEPS)
    # Set USE_PER_SCENARIO_HORIZON=False (or pass max_steps_per_episode in env config) to use
    # the fixed EPISODE_STEPS instead.
    USE_PER_SCENARIO_HORIZON = True
    HORIZON_STEP_MARGIN = 6    # extra steps so the last aircraft can land and settle (~270 s)
    MIN_EPISODE_STEPS = 48     # floor on the per-scenario horizon
    MAX_EPISODE_STEPS = 130    # cap (bounds per-step rollout cost)

    # ------------- Optimizer configuration (PB2) -------------
    LR_MIN, LR_MAX = 3e-04, 3e-4  # Learning rate
    ENTROPY_MIN, ENTROPY_MAX = 0.005, 0.1  # Entropy coeff
    CLIP_MIN, CLIP_MAX = 0.1, 0.4  # PPO clip parameter

    # ------------- Environment configuration -------------
    MAX_AIRCRAFT_COUNT_TRAIN = 10  # The maximum number of aircraft used during training
    MAX_AIRCRAFT_COUNT_EVAL = 10  # The maximum number of aircraft used during evaluation
    MAX_WAYPOINTS = 20  # The maximum number of waypoints in flight plan for single aircraft
    AIRCRAFT_SCALAR_LEN = 12  # Number of features in aircraft scalar vector (incl. time-to-conflict)
    # AMAN-sequence features (windowed env): 4 extra per-aircraft columns appended after the 12
    # above -- predicted-vs-AMAN rank shift, predicted crossings, and the landing-gap error to the
    # AMAN predecessor and successor (see WindowedATCEnv._sequence_state). Env-overridable for the
    # same reason as ACTION_SET: it sizes the observation space and the network's first layer, and
    # SubprocVecEnv workers re-import this module, so it must be in the environment, not set in
    # main(). The 10-aircraft env leaves the extra columns at zero.
    #     TADA_SEQUENCE_OBS=1 python main_jax.py --env windowed ...
    SEQUENCE_OBS = os.getenv("TADA_SEQUENCE_OBS", "0") == "1"
    SEQUENCE_FEATURES = 4
    if SEQUENCE_OBS:
        AIRCRAFT_SCALAR_LEN += SEQUENCE_FEATURES
    # Reselection (windowed env): up to MAX_PICKS clearances per 45 s step. The action gains a
    # third component, "again": the policy's own request for another pick before the clock
    # advances. The next pick is made on a fresh observation whose prediction includes the
    # clearances already queued, so it sees their effect. 1 = off: the action stays (aircraft,
    # clearance) and nothing changes. Env-overridable like SEQUENCE_OBS (it sizes the spaces and
    # adds a per-aircraft "cleared at this step" column, appended last).
    #     TADA_MAX_PICKS=2 python main_jax.py --env windowed --reward-mode outcome_pbrs ...
    MAX_PICKS = int(os.getenv("TADA_MAX_PICKS", "1"))
    # Select the observation frame used for geometric clearance legality.
    MASK_FRAME_FIX = os.getenv("TADA_MASK_FRAME_FIX", "1") == "1"
    # Select the turn direction convention used by clearance legality.
    TURN_MASK_FIX = os.getenv("TADA_TURN_MASK_FIX", "0") == "1"
    PICK_FEATURES = 1
    if MAX_PICKS > 1:
        AIRCRAFT_SCALAR_LEN += PICK_FEATURES
    # Initial logit of again=1 for a fresh "again" head (sigmoid(-3) ~ 5%): a warm-started policy
    # starts close to its one-pick behaviour and explores the second pick occasionally.
    AGAIN_INIT_LOGIT = -3.0
    FLIGHT_PLAN_WAYPOINT_DIM = (
        11  # Number of features in flight plan waypoint vector (includes global)
    )
    SUCCESS_DELAY_THRESHOLD = 60.0
    # Realised losses and near conflicts use separate severity and distance thresholds.
    DEVIATION_OFFSET = 0.0  # Offset for time deviation. Set to 0.0 to disable.

    # ------------- Simulator configuration -------------
    # Simulation interval: seconds of sim time advanced per env step == how often the agent may
    # issue a clearance (the "advance steps per environment step call").
    TIME_BETWEEN_ACTIONS = 45  # Time between actions in seconds
    SIMULATOR_CONFIG_ROOT = os.getenv("TADA_SIMULATOR_CONFIG_ROOT", "./config/simulation_config/")
    # Shared generator uses SIMULATOR_CONFIG_ROOT and rejection; procedure mode uses
    # procedure-specific snapshots and preserve each sampled scenario.
    SCENARIO_GENERATOR = os.getenv("TADA_GENERATOR", "procedure").lower()
    if SCENARIO_GENERATOR not in ("procedure", "shared"):
        raise ValueError(f"TADA_GENERATOR={SCENARIO_GENERATOR!r}: expected 'procedure' or 'shared'")
    PROCEDURE_CONFIG_ROOTS = {  # use case -> generator snapshot (README.md in each)
        1: "./config/simulation_config_mxp/",
        2: "./config/simulation_config_bgy/",
    }
    ORIGIN_LAT = 60.0  # Origin latitude for scenario recording
    ORIGIN_LON = 25.0  # Origin longitude for scenario recording

    # ------------- Scenario generation configuration -------------
    WORLD_JSON_FILE = None  # Path to the world file to override scenario generation
    DEFAULT_TRAINING_DIFFICULTY = MissionDifficulty.VALIDATION_USE_CASE_2  # Difficulty for train
    DEFAULT_EVALUATION_DIFFICULTY = MissionDifficulty.VALIDATION_USE_CASE_2  # Difficulty for eval
    # Evaluation seed pool. Defaults to _DEFAULT_EVAL_SEEDS, overridden by config/eval_seeds.txt
    # if present (see _load_eval_seeds). evaluation cadence governs how many of these a
    # single eval pass covers; with fewer episodes than seeds the pass walks a rolling window and
    # the full pool is covered across passes (the *_rolling eval metrics aggregate it).
    SCENARIO_EVAL_SEEDS = _load_eval_seeds(EVAL_SEEDS_FILE, _DEFAULT_EVAL_SEEDS)
    SCENARIO_TRAIN_SEED = None  # Use for debugging purposes only. Set to None to disable.
    SCENARIO_INITIAL_CONFLICT_FREE_STEPS = 6  # Number of steps required to be conflict-free (~270 s)
    # Bound scenario resampling when the rejection filter cannot find a usable scenario.
    SCENARIO_MAX_ATTEMPTS = 50

    # ------------- Neighbor observation/encoder configuration -------------

    # ------------- Conflict detection configuration -------------
    # Detection and severity use distinct radius pairs.
    #
    #   DETECTION_* : how far out the simulator RAISES a LOSS_OF_SEPARATION event, and how far
    #                 the observation can watch one develop. Wide on purpose — the agent needs
    #                 early warning about pairs it will never actually be charged for.
    #   SEVERITY_*  : the band over which severity ramps 0 -> 1. Narrow on purpose — severity
    #                 1.0 must mean a real loss of separation (<= 3 NM AND <= 500 ft), and legal
    #                 spacing (>= 5 NM) must cost exactly nothing.
    #
    DETECTION_SAFEZONE_RADIUS_NM = 10.0   # sim reporting radius + observation ramp outer edge
    DETECTION_CONFLICT_RADIUS_NM = 3.0    # observation ramp saturates here (the LoS boundary)
    SEVERITY_SAFEZONE_NM = 5.0            # severity reaches 0 here
    SEVERITY_CONFLICT_NM = 3.0            # severity reaches 1.0 here — the LoS boundary
    # The Mission's horizontal safezone radius. The simulator raises LOSS_OF_SEPARATION out to
    # TWICE this, and the AMAN spaces landings by it (see SCENARIO_GENERATOR). At 5 NM the events
    # reach exactly DETECTION_SAFEZONE_RADIUS_NM, so the observation and severity are unchanged.
    MISSION_SAFEZONE_RADIUS_NM = 5.0 if SCENARIO_GENERATOR == "procedure" else DETECTION_SAFEZONE_RADIUS_NM
    VERTICAL_SAFEZONE_RADIUS_FT = 1000.0  # Vertical safezone radius in feet
    VERTICAL_CONFLICT_RADIUS_FT = 500.0   # Vertical conflict radius in feet

    # Shape of the severity decay across the SEVERITY_CONFLICT_NM -> SEVERITY_SAFEZONE_NM band,
    # applied ONCE to the combined (horizontal, vertical) normalised separation:
    #     severity = u ** p      where u in [0,1] is 1 at the conflict radius, 0 at the safezone
    # p = 1.0 reproduces the linear ramp exactly (free ablation). p = 2.0 is convex: it puts the
    # strongest restoring gradient immediately outside 3 NM (0.95/NM at 3.1 NM vs 0.50 linear)
    # and joins the zero-penalty region at 5 NM with matching slope, so there is no kink for the
    # policy to sit against at legal spacing. The corner is at 3 NM, which is where we *want* a
    # hard edge — the episode ends there anyway.
    SEVERITY_SHAPE_P = 2.0

    # Severity at/above which a pair counts as a REALISED loss of separation. The ramp is
    # clamped, so 1.0 is attained exactly at (<= 3 NM AND <= 500 ft); compare with a small
    # epsilon rather than `== 1.0`. This is the single failure definition: it drives the tier
    # gate, `end_reason="separation"`, and episode termination.
    LOS_SEVERITY_THRESHOLD = 1.0

    # "Near conflict" for the eval shield and the ever_near_conflict diagnostic. Pinned as a
    # DISTANCE, not a severity: a severity threshold silently changes meaning whenever
    # SEVERITY_SHAPE_P or the band edges move (0.7 is 3.60 NM at p=1 but 3.33 NM at p=2).
    NEAR_CONFLICT_DISTANCE_NM = 3.5

    # ------------- Action configuration -------------
    # TMB and PMS use distinct 15-clearance sets, selected at import time.
    ACTION_SET = os.getenv("TADA_ACTION_SET", "v2").lower()
    _ACTION_SET_SIZES = {"v2": 15, "v2pms": 15}
    # Optional per-procedure blocking mask over the set (actions/action_set.py AIRPORT_MASKS), e.g.
    # TADA_AIRPORT_MASK=procedure for per-airport masks. Import-time, like ACTION_SET.
    AIRPORT_MASK = os.getenv("TADA_AIRPORT_MASK", "").lower() or None
    # Stated here as a literal because config cannot import the action modules without a
    # cycle; actions/action_set.py asserts it matches len(Action) at import time.
    NUM_ACTIONS = _ACTION_SET_SIZES[ACTION_SET]  # clearances available to the agent
    # In single-agent autoregressive setting, the number of clearances equals NUM_ACTIONS
    N_CLEARANCES = NUM_ACTIONS

    # Reserved for a future continuous-magnitude head (ASTRA-style 3rd head): a normalized
    # magnitude sampled from a Gaussian after the discrete (aircraft, clearance TYPE) choice.
    # Not wired yet; the current autoregressive path keeps clearances fully discrete.
    ACTION_HEADING_ADJUSTMENT = 40.0  # The maximum heading adjustment in degrees during vectoring
    ACTION_SPEED_ADJUSTMENT_SMALL = 10.0  # The maximum speed adjustment in knots
    ACTION_SPEED_ADJUSTMENT_MEDIUM = 20.0  # v2 only: interpolates SMALL and LARGE
    ACTION_SPEED_ADJUSTMENT_LARGE = 30.0  # The maximum speed adjustment in knots
    WAYPOINTS_TO_LEAVE_AT_MINIMUM = 4  # Minimum waypoints to keep when skipping
    MAX_TURN_ANGLE_AFTER_HEADING_ADJUSTMENT = 90.0  # Turn angle to return to route after vectoring
    SKIP_MAX_INTERCEPT_ANGLE_DEG = 95.0  # Max angle to intercept remaining route after skip
    # Optional custom action mask, 1 = action is allowed, 0 = action is not allowed. Can be set None
    CUSTOM_ACTION_MASK = None #"[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1] #[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]

    # ------------- Single-agent autoregressive configuration -------------
    # Max aircraft indices exposed in the Tuple(Discrete(N_aircraft), Discrete(N_clearances))
    N_AIRCRAFT_MAX = MAX_AIRCRAFT_COUNT_EVAL  # action space (N_aircraft, N_clearances)
    GLOBAL_DIM = 4  # Global feature vector size
    AR_HIDDEN = 128  # Hidden size for autoregressive module encoders/heads

    # The context fed to every head is a masked MEAN over aircraft: permutation invariant, and
    # structurally unable to represent a PAIRWISE relation. "Aircraft i and j are converging" is
    # the core relation in a separation problem and it has to be reconstructed from an average.
    # the per-aircraft embeddings already sit at mean cosine 0.70 before the mean averages them.
    # One masked self-attention block over the aircraft slots lets slot i read slot j directly.
    # Self-attention with a key mask is permutation EQUIVARIANT, so the property verified in
    USE_AIRCRAFT_ATTENTION = True
    ATTENTION_HEADS = 4
    ATTENTION_DIM = 64          # low-rank q/k/v; keeps the added cost to ~33k params
    CONV1_CHANNELS = 64
    CONV2_CHANNELS = 64

    # ------------- Normalization configuration -------------
    GLOBAL_POS_XY_MAX = 150.0  # in nautical miles (same for X and Y)
    GLOBAL_POS_Z_MAX = 45000.0  # in feet (altitude range)

    RENDER_POS_XY_MIN, RENDER_POS_XY_MAX = -100.0, 100.0  # for rendering only

    # Relative altitude normalization window (+/- value in feet)
    REL_POS_Z_RANGE_FT = 40000.0

    # Velocity normalization (in knots and feet per minute)
    VEL_XY_MAX = 350.0  # knots (same for X and Y horizontal velocity)
    VEL_Z_MAX = 3000.0  # feet per minute (vertical velocity)

    # Relative feature normalization ranges
    # REL_VEL_RANGE_KTS = 500.0
    # REL_VZ_RANGE_FPM = 2000.0  # vertical speed difference range (feet per minute)
    # TTCA_MAX_S = NEIGHBOR_LOOKAHEAD_MULTIPLIER * TIME_BETWEEN_ACTIONS  # Time to collision ahead
    # DCPA_MAX_NM = (
    #     NEIGHBOR_LOOKAHEAD_MULTIPLIER * TIME_BETWEEN_ACTIONS * VEL_XY_MAX / 3600.0
    # )  # Distance to collision ahead

    # Signed-log deviation gives more resolution near zero.
    TIME_DEVIATION_MAX = 1800  # max |deviation| represented before saturating (s)
    TIME_DEVIATION_LOG_FLOOR_S = 10.0  # log knee; smaller = more resolution near zero
    USE_LOG_DEVIATION_OBS = True  # False restores the linear encoding (free ablation)

    # Normalize time to target over the episode horizon with signed-log resolution.
    TIME_TO_TARGET_MIN = 0
    TIME_TO_TARGET_MAX = EPISODE_STEPS * TIME_BETWEEN_ACTIONS  # 2880 s
    TIME_TO_TARGET_LOG_FLOOR_S = 30.0

    # Predicted infringement normalization
    PREDICTED_INFRINGEMENT_MIN, PREDICTED_INFRINGEMENT_MAX = 0.0, 1.0  # Predicted infringement

    MAX_HISTORY = 8 # Maximum action history in the observables
    # ------------- Reward configuration -------------
    # Maximum points awarded for fully mitigating the delay to zero (per episode upper bound)
    # The reward function grants incremental progress so that reaching zero delay yields at most
    # this many points in total (before action penalties).
    REWARD_SCALE_DEVIATION = 2000 # Divides total deviation estimate (lowered 4000->2000 to give deviation real gradient vs conflict)

    REWARD_ACTION_PENALTY_WEIGHT = 0.02  # Small per-command cost; INCLUDED in reward total()
    # Degrees-of-freedom scaling of the action cost: acting late (few remaining waypoints)
    # is more expensive than acting early. Linear ramp on the commanded aircraft's remaining
    # waypoint count r:  factor = 1 + K * (W - r) / W  (W = MAX_WAYPOINTS), clamped >= 1.
    # K=0 disables (flat cost). e.g. K=3, W=20: r=20->1.0x, 10->2.5x, 4->3.4x, 1->3.85x.
    REWARD_ACTION_DOF_K = 3.0
    REWARD_INFRINGEMENT_WEIGHT = 10.0  # Conflict scale; with the ramp below, imminent conflicts dominate

    # --- Imminence-weighted conflict decay ---
    # A predicted conflict is discounted by how far in the future it is:
    #     w(ttc) = max(CONFLICT_FAR_FLOOR, 2 ** (-ttc / CONFLICT_HALF_LIFE_S))
    #
    # The half-life approximates the time a clearance needs to affect spacing.
    CONFLICT_HALF_LIFE_S = 240.0
    CONFLICT_FAR_FLOOR = 0.005          # numerical floor so very distant conflicts stay non-zero
    NEAR_CONFLICT_TIME_S = 1200.0       # 20 min window for has_near_conflict() (eval shield only,
                                        # NOT used by the dense reward any more)

    # --- Landing-proximity deviation weighting ---
    # Each aircraft's |deviation| is weighted linearly from this baseline (far from
    # landing) up to 1.0 (landing imminent).
    REWARD_DEVIATION_LANDING_BASELINE = 0.2

    # --- Success criteria ---
    # The binary "solved" flag is now tier 3 of the graduated bonus below
    # (all landed + every aircraft within SUCCESS_TIER2_DEV_S + no near-conflict).
    # Relaxed near-conflict gate: conflicts are tolerated earlier in the episode but must be
    # cleared for the final K steps (i.e. resolved "by the time the aircraft lands"), rather
    # than the unsatisfiable "no near-conflict ever".
    SUCCESS_NEAR_CONFLICT_CLEAR_STEPS = 3  # ~135 s clear window

    # --- Realised-only conflict gate ---
    # True  => the tier gate counts only conflicts that ACTUALLY MATERIALISED on the live
    #          world: some pair was within HORIZONTAL_CONFLICT_RADIUS_NM..3 NM (severity >=
    #          NEAR_CONFLICT_THRESHOLD) at a real timestep. This matches the operational
    #          definition of a loss of separation.
    # False => the gate also trips on conflicts merely PREDICTED by the
    #          NOOP rollout within NEAR_CONFLICT_TIME_S, so an episode that was safe in
    #          reality could still score tier 0 on a pessimistic forecast.
    #
    # Note this only relaxes the SUCCESS/TIER gate. The dense conflict penalty in
    # actions/rewards.py::infringement_reward_from_world still runs on the predicted world,
    # so the agent keeps its incentive to head conflicts off before they happen.
    SUCCESS_CONFLICT_REALISED_ONLY = True

    # --- Graduated success bonus tiers (terminal; highest tier reached wins, no stacking) ---
    # All tiers also require the relaxed no-near-conflict gate above. Five tiers guide the
    # agent progressively from "most aircraft roughly on time" to "solved".
    #   T1: >= SUCCESS_TIER1_FRAC (50%) within SUCCESS_TIER1_DEV_S (±120 s)
    #   T2: >= SUCCESS_TIER2_FRAC (80%) within SUCCESS_TIER2_DEV_S (±100 s)
    #   T3: >= 80% within SUCCESS_TIER3_DEV_S (±70 s) AND max_aircraft_dev < TIER3_MAX_DEV_S (200 s)
    #   T4: >= 80% within SUCCESS_TIER4_DEV_S (±60 s) AND max_aircraft_dev < TIER4_MAX_DEV_S (120 s)
    #   T5: all landed AND all aircraft within SUCCESS_TIER5_DEV_S (±60 s)  — "solved"
    # T3/T4 cap the worst-aircraft deviation; each tier implies all lower tiers.
    SUCCESS_TIER1_DEV_S = 120.0
    SUCCESS_TIER1_FRAC  = 0.5
    SUCCESS_TIER2_DEV_S = 100.0
    SUCCESS_TIER2_FRAC  = 0.8       # also used for T3 and T4 fraction requirements
    SUCCESS_TIER3_DEV_S = 70.0
    SUCCESS_TIER3_MAX_DEV_S = 200.0
    SUCCESS_TIER4_DEV_S = 60.0
    SUCCESS_TIER4_MAX_DEV_S = 120.0
    SUCCESS_TIER5_DEV_S = 60.0      # same threshold as T4 but requires *all* aircraft
    REWARD_SUCCESS_BONUS_TIER1 = 1.5
    REWARD_SUCCESS_BONUS_TIER2 = 3.0
    REWARD_SUCCESS_BONUS_TIER3 = 6.0
    REWARD_SUCCESS_BONUS_TIER4 = 9.0
    REWARD_SUCCESS_BONUS_TIER5 = 13.0

    # --- Terminal penalty for a realised hard violation (loss of separation / airspace exit) ---
    # A realised violation terminates the episode. The forfeit term discourages
    # early termination from becoming profitable.
    REWARD_VIOLATION_PENALTY = 30.0            # BASE
    REWARD_VIOLATION_PER_REMAINING_STEP = 0.5  # forfeit-the-horizon term; 0.0 disables

    # Strict-domination mode (OFF by default). When set to a positive value, a violating
    # episode's TOTAL return is clamped to exactly -this, whatever it had banked beforehand.
    # That is literally "the worst possible outcome": every loss of separation scores the
    # same, worst score, so there is no timing incentive and no bank-then-crash strategy.
    #
    # The cost is real and is why it is not the default: the terminal reward then depends on
    # the return accumulated so far, which is NOT in the observation, so the MDP stops being
    # Markovian and the critic cannot predict the terminal value. Expect value loss to rise
    # and explained_variance to fall. The bounded scheme above gets the same ORDERING
    # (a bust always scores below a comparable clean episode) while staying predictable.
    #
    # Achieving strict domination through BASE alone would need ~70 — the gap between the
    # best clean return (~+44) and the worst (~-30) — which is a large enough spike to
    # destabilise reward normalisation on its own.
    REWARD_VIOLATION_RETURN_FLOOR = 0.0

    # --- Goal-closeness bonus (dense, potential-based shaping) ---
    # Positive saturating bonus that grows as predicted deviations approach zero.
    # Jointly penalises total deviation AND the worst aircraft, providing per-step
    # gradient signal on both the aggregate schedule and the tail aircraft:
    #   bonus = MAX * exp(-total_abs_dev / SCALE_TOTAL) * exp(-max_aircraft_dev / SCALE_MAX)
    # Set MAX = 0 to disable. SCALE_MAX = 0 disables the max-dev factor.
    REWARD_GOAL_BONUS_MAX = 2.0
    REWARD_GOAL_BONUS_SCALE = 400.0      # seconds; controls total_abs_dev sensitivity
    REWARD_GOAL_BONUS_SCALE_MAX = 200.0  # seconds; controls max_aircraft_dev sensitivity

    # --- Potential-Based Reward Shaping (PBRS) tier potential ---
    # Distributes the tier-ladder signal densely across steps so the critic does not
    # need to predict a large terminal lump.  γ MUST match PPO's GAMMA in main.py.
    PBRS_GAMMA = 0.998          # must equal main.py::GAMMA
    PBRS_KAPPA = 0.1            # sigmoid smoothness for tier-fraction axis (fraction units)
    PBRS_KAPPA_MAX_DEV = 30.0   # sigmoid smoothness for max_aircraft_dev axis (seconds)

    # Coverage on the tier axis is a FRACTION OF AIRCRAFT, so with 10 aircraft it only takes
    # 11 values. Smoothing a variable that lives on an 11-point lattice does not make Φ smooth:
    # measured, Φ is a staircase and the per-step shaping is a spike each time one aircraft
    # contributes a SOFT membership σ((d_j − |dev_i|)/κ_dev) instead of a hard 0/1, so Φ responds
    # to any change in any aircraft's deviation. Φ stays a function of state alone, so the
    # policy-invariance guarantee (Ng et al. 1999) is untouched.
    #
    # Tier awards use hard counts; the potential uses smooth coverage.
    PBRS_CONTINUOUS_COVER = True
    PBRS_KAPPA_DEV = 15.0       # seconds; width of the per-aircraft membership sigmoid
    # Small terminal confirmation for Tier 5 only; replaces the large tier-ladder cliff.
    REWARD_SUCCESS_BONUS_TIER5_CONFIRM = 1.0
