"""Configuration for the WINDOWED single-agent env (atc_env/windowed_env.py).

Values not listed here are inherited from config.Config, including action sets,
observation encodings, conflict bands and action costs.

Design rule: nothing the agent OBSERVES, and nothing its value function has to PREDICT, may
depend on how many flights the scenario has. A 20-flight training scenario is treated as a
finite excerpt of an endless arrival stream, so the same agent can be run on a longer stream
at evaluation (a stability test for continuous use) without any input leaving its range.
"""

from config.config import Config
from flight_simulator import MissionDifficulty


# pylint: disable-next=too-few-public-methods
class WindowedConfig:
    """Windowed-env settings. Read by WindowedATCEnv only."""

    # ------------- Scenario -------------
    # Pinned per env, independent of Config.DEFAULT_*_DIFFICULTY, so switching the default env
    # between trombone and point-merge never silently changes the windowed experiment.
    DIFFICULTY = MissionDifficulty.VALIDATION_USE_CASE_1  # MXP trombone
    SCENARIO_AIRCRAFT_COUNT = 20  # flights generated per scenario (overridable per env)

    # Stitching: build a stream from STITCH_SEGMENTS independently generated scenarios of
    # SCENARIO_AIRCRAFT_COUNT flights each (Simulator._stitched_scenario). Each segment's first
    # target landing follows the previous segment's last one after a random cooling gap drawn
    # uniformly from STITCH_GAP_S. Generating 40 flights in ONE sequence instead lets the
    # generator's schedule pressure build along the queue (do-nothing |dev| ~220 s -> ~910 s),
    # which is a backlog test, not a steady-state one. The random gap also varies the traffic
    # pattern the policy sees at every segment seam. 1 = a single generated scenario.
    STITCH_SEGMENTS = 1
    STITCH_GAP_S = (120.0, 900.0)

    # ------------- Window -------------
    # Slots the agent sees. MUST equal Config.N_AIRCRAFT_MAX: that keeps the observation and
    # action spaces identical to the 10-aircraft env, so an existing checkpoint (same
    # ACTION_SET) can warm-start the windowed agent without surgery.
    WINDOW_SIZE = int(Config.N_AIRCRAFT_MAX)

    # The window is the next WINDOW_SIZE flights in the landing QUEUE (AMAN target order;
    # measured: the do-nothing landing order equals the AMAN order). When a flight lands, the
    # next flight in the queue takes its place, so the window stays full until the stream runs
    # dry.
    #   False -- queue membership only: a flight that has not spawned yet still holds its place
    #            (its under_control feature is 0). This is the "replace with the next in queue"
    #            semantics.
    #   True  -- only flights already under control are eligible (the 10-aircraft env's rule).
    WINDOW_UNDER_CONTROL_ONLY = False
    # Not-yet-spawned flights in the window are visible (the agent can plan around them) but
    # only DO_NOTHING is unmasked for them: a clearance to an aircraft that is not under control
    # has no operational meaning.
    PENDING_ACTIONABLE = False

    # ------------- Time encoding -------------
    # Scale for the window's time features (per-flight scheduled time to go, and the window
    # span). The shared TIME_TO_TARGET_MAX (2880 s) is one flight's horizon; ten flights at the
    # generator's 90-600 s landing spacing span up to ~6000 s, so the base scale pinned the
    # span at +1 a quarter of the time. Same signed-log shape and knee as the base encoding.
    TIME_SCALE_S = 7200.0

    # ------------- Prediction rollout (receding horizon) -------------
    # The base env rolls out to the scenario's LAST landing every step, so the rollout's length,
    # and every feature read from its end state, depends on scenario size. Here it covers
    # max(fixed lookahead, the window's last scheduled landing) plus a margin for late flights:
    # bounded by the window, independent of the scenario.
    ROLLOUT_LOOKAHEAD_S = float(Config.EPISODE_STEPS * Config.TIME_BETWEEN_ACTIONS)  # 2880 s
    ROLLOUT_MARGIN_STEPS = 20  # 900 s: a flight up to 15 min late still lands inside the rollout
    ROLLOUT_MAX_STEPS = 200

    # ------------- Episode horizon (truncation only) -------------
    # 20 flights at ~220 s spacing put the last AMAN ETA at 5000-7100 s (111-158 steps).
    MAX_EPISODE_STEPS = 200
    MIN_EPISODE_STEPS = 48
    HORIZON_STEP_MARGIN = 6

    # Continuing-task semantics. The stream running dry (every flight landed) or the step cap
    # is a TRUNCATION, not a termination: PPO bootstraps the value there instead of treating it
    # as an absorbing end. Otherwise V(s) would have to encode "how many flights are left in
    # this scenario", which is not observable. A realised loss of separation / airspace exit
    # stays a true termination.
    CONTINUING_TASK = True

    # ------------- Reward: per flight, not per episode -------------
    # The tier ladder is a statement about a FIXED SET ("80% of the 10 aircraft within 60 s"),
    # and the PBRS potential built on it inherits that. The windowed env scores each FLIGHT
    # once, at the step it lands, from its REALISED landing deviation:
    #
    #   landing_bonus(dev) = sum_k  B_k * sigmoid((D_k - |dev|) / KAPPA)
    #
    # Two rungs: a coarse one (120 s) so rough landings still earn something, and the success
    # rung (60 s = SUCCESS_TIER5_DEV_S). 3.0 per flight ~ the 10-aircraft ladder's 32.5 total
    # spread over 10 flights, so per-flight incentives match the agent we warm-start from.
    LANDING_BONUS_RUNGS = ((120.0, 1.0), (60.0, 2.0))  # (D_k seconds, B_k)
    LANDING_BONUS_KAPPA_S = 15.0

    # Dense deviation term: every not-yet-landed flight whose predicted landing is within
    # DEVIATION_RAMP_S, weighted linearly from DEVIATION_FAR_WEIGHT at the ramp edge to 1.0 at
    # touchdown. With a far weight of 0 a flight's charge fades in smoothly as it approaches
    # rather than appearing at full size when it enters the window or the rollout, and landed
    # flights are never charged again (a perpetual per-step charge would dominate a long
    # stream). The ramp is <= ROLLOUT_LOOKAHEAD_S, so every flight on it is inside the rollout.
    #   "unlanded" -- as above. "window" -- window flights only. "all" -- 10-aircraft behaviour.
    DEVIATION_SCOPE = "unlanded"
    DEVIATION_RAMP_S = ROLLOUT_LOOKAHEAD_S
    DEVIATION_FAR_WEIGHT = 0.0

    # Dense conflict term: pairs with at least one member in the window (a pair the agent can
    # neither see nor command is noise). "all" is the 10-aircraft behaviour. A realised loss of
    # separation still terminates the episode wherever it happens.
    CONFLICT_SCOPE = "window"

    # Absorbing-failure penalty, CONSTANT: BASE plus a window's worth of forfeited landings. A
    # penalty scaled by flights (or steps) remaining in the scenario is unobservable and would
    # make the critic's terminal target depend on scenario length.
    VIOLATION_PENALTY_BASE = float(Config.REWARD_VIOLATION_PENALTY)
    VIOLATION_FORFEIT = float(sum(b for _, b in LANDING_BONUS_RUNGS)) * WINDOW_SIZE

    # ------------- Reward mode -------------
    # "legacy"       -- dense per-step reward (above).
    # "outcome_pbrs" -- the reward is the OBJECTIVE only, and everything else is potential-based
    #                   shaping (Ng et al. 1999), which cannot change which policy is optimal:
    #
    #   r = sum over flights landing this step of landing_value(|dev|)   (bracket step function)
    #       - CLEARANCE_COST * [a real clearance, not DO_NOTHING]
    #       - PBRS_VIOLATION_PENALTY * [realised loss of separation / airspace exit]
    #       + gamma * Phi(s') - Phi(s)                      (Phi(s') = 0 after a bust)
    #
    # Potential shaping measures changes in the predicted state while the objective
    # scores outcomes and real clearances.
    REWARD_MODE = "legacy"

    # Objective, LEXICOGRAPHIC: safety, then each flight's deviation BRACKET, then the number of
    # clearances. An optimal ATFM policy meets the first two with the fewest actions, so two
    # solutions with the same safety and the same brackets must differ in score only by how many
    # clearances they issued -- never by where inside a bracket a flight landed.
    #
    # landing_value(|dev|) is a step function of the bracket (paid once, at touchdown):
    LANDING_BRACKETS = ((60.0, 3.0), (120.0, 1.0), (300.0, 0.0), (600.0, -1.5))  # (|dev| <= s, value)
    LANDING_VALUE_BEYOND = -3.0  # |dev| > 600 s: a badly late flight is never free
    # The hard-solved target (every flight within +/-30 s): the same ladder with a top rung one full
    # bracket above +/-60 s, so the steps stay >= 1.0 and the clearance cost still cannot buy a worse
    # bracket. Selected per env (config key "landing_brackets"; main_jax.py --hard-bracket). The bust
    # penalty follows from the ladder: BASE + (4 - (-3)) x WINDOW_SIZE = 100.
    HARD_BRACKETS = ((30.0, 4.0),) + LANDING_BRACKETS
    # Clearances: a flat cost per REAL clearance (DO_NOTHING free; no late-clearance factor -- the
    # criterion is the count). Sized for the WORST case: at most one clearance per 45 s step and
    # a 20-flight stream runs up to ~160 steps, so 160 x 0.006 = 0.96 < 1.0, the smallest gap
    # between brackets. Saving actions can then never buy a worse bracket, and between equal
    # brackets fewer actions always score higher. (Calibrated for 20-flight streams: with a
    # constant per-action cost, strict ordering cannot hold for arbitrarily long streams.)
    CLEARANCE_COST = 0.006
    # Safety dominates: a loss of separation costs more than landing the whole visible window at
    # the worst bracket instead of the best: BASE + (3 - (-3)) x WINDOW_SIZE = 30 + 60 = 90 (plus
    # the rest of the stream, which it forfeits).
    PBRS_VIOLATION_PENALTY = float(Config.REWARD_VIOLATION_PENALTY) + (
        LANDING_BRACKETS[0][1] - LANDING_VALUE_BEYOND) * WINDOW_SIZE
    # The deviation POTENTIAL uses the brackets smoothed with sigmoids of this width, so the
    # shaping still has a gradient inside a bracket; potentials cannot change what is optimal.
    LANDING_SMOOTH_KAPPA_S = 15.0

    # Potential Phi(s), from the do-nothing prediction (a deterministic function of the state):
    #   Phi_dev  = + sum over unlanded flights of w * smoothed landing_value(|predicted dev|)
    #              i.e. the predicted value of the objective; w ramps 0 -> 1 over the last
    #              DEVIATION_RAMP_S before the flight's predicted landing
    #   Phi_conf = - PHI_CONFLICT_WEIGHT * sum of predicted-conflict severity x imminence
    #              (window-scoped predicted conflicts as a potential)
    #   Phi_seq  = - PHI_SEQ_PER_SWAP * sum over predicted AMAN-order inversions of w
    #   Phi_gap  = - PHI_GAP_PER_MIN * sum over AMAN-consecutive pairs of
    #              w * max(0, PHI_GAP_MIN_S - predicted landing gap) / 60
    # gamma is Config.PBRS_GAMMA, which must equal PPO's.
    PHI_CONFLICT_WEIGHT = float(Config.REWARD_INFRINGEMENT_WEIGHT)
    # With "pbrs_flat_potentials" (no imminence discount) a predicted conflict counts in full
    # however far out. At the ramped weight (10) creating or resolving one far-off conflict would
    # then outweigh several flights' landing values; 1.0 makes a predicted loss worth about one
    # bracket step, so the objective's scale is kept and the bust penalty still rules safety.
    PHI_CONFLICT_WEIGHT_FLAT = 1.0
    PHI_SEQ_PER_SWAP = 1.0
    PHI_GAP_MIN_S = 90.0  # the AMAN minimum landing spacing (scenario_generator/config.toml)
    PHI_GAP_PER_MIN = 1.0

    # ------------- Success (reporting only, never rewarded) -------------
    ON_TIME_DEV_S = float(Config.SUCCESS_TIER5_DEV_S)  # a flight is "on time" within +/-60 s
