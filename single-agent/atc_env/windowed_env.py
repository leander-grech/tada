"""Windowed single-agent ATC env: a stream of arrivals seen through a window of the next flights.

A scenario has WindowedConfig.SCENARIO_AIRCRAFT_COUNT flights (20). The agent sees WINDOW_SIZE
(10) of them: the next flights in the landing queue (AMAN target order). When one lands it
leaves the window and the next flight in the queue takes its place.

The scenario is treated as a finite excerpt of an endless arrival stream: nothing the agent
observes, and nothing its critic has to predict, depends on how many flights the scenario has.
That is what lets the same agent be run on a longer stream at evaluation as a test of whether
it can be used continuously.

Inherited unchanged from ATCSingleAgentEnv: observation and action spaces (same slot count, so
existing checkpoints load), the action set and its interpretation, conflict detection,
realised-LoS termination, and the tier ladder -- still COMPUTED over all flights at episode end
for reporting, but no longer part of the reward.

What changes, and why (numbers in config/windowed_config.py):
  * slots hold the window in queue order, not "every active aircraft in world order";
  * the NOOP prediction rollout is a receding horizon bounded by the window, not "until the
    scenario's last landing";
  * observables that encoded the scenario rather than the window are replaced (see
    _build_window_observation);
  * the reward is per FLIGHT, paid when it lands, instead of the per-EPISODE tier ladder + PBRS;
  * dense deviation / conflict terms are scoped (flights approaching landing / pairs touching
    the window);
  * continuing-task semantics: the stream running dry is a truncation, and the bust penalty
    is a constant.

Reselection (Config.MAX_PICKS > 1, off by default): the action is (aircraft, clearance, again).
again=1 queues the clearance WITHOUT advancing the clock and asks the policy for another pick,
on a new observation whose prediction includes every clearance queued so far. again=0 applies
the queued clearances together with this one and advances 45 s, as a one-pick step does. The
env never offers a pick: only the policy asks. Masks (see _build_window_observation):
  * mask_select: under-control slots not yet cleared at this step (the old rule otherwise);
  * an aircraft cleared at this step can only be left alone for the rest of it;
  * mask_again: another pick is possible only with budget left (MAX_PICKS per step) and at
    least one other clearable aircraft; the policy also forces again=0 with DO_NOTHING.
Returns: a queued pick pays its clearance cost and the change in the potential at once, and
takes no time, so it is discounted by 1 (info["zero_time"]); jax_ppo/ppo.py uses that
per-transition discount and the shaping below uses the same one, so it still telescopes.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from actions.actions import Action, interpret_action
from actions.rewards import (
    _action_penalty,
    _deviation_reward_rollout,
    get_aircraft_time_deviations,
    has_near_conflict,
    infringement_reward_from_world,
)
from atc_env.single_agent_env import ATCSingleAgentEnv
from config.config import Config
from config.simulation_config_path import get_config_path
from config.windowed_config import WindowedConfig
from flight_simulator import Aircraft, MissionDifficulty, simulate_n_next_states
from gymnasium import spaces
from matplotlib.colors import to_rgba
from models.observations import (
    AircraftScalarFeatures,
    _map01_to_sym,
    build_global_observation,
    map_deviation,
    map_signed_log,
    map_to_sym_unit,
)
from models.commands import TimedCommandList
from simulator.simulator import Simulator
from utils.reward_utils import RewardParts

# Column indices into obs["aircraft"] / obs["global"], derived from the feature definitions so
# they cannot silently drift if a feature is inserted.
_AC_FIELDS = list(AircraftScalarFeatures.__dataclass_fields__)
_AC_TTT = _AC_FIELDS.index("time_to_target")
_AC_PIN = _AC_FIELDS.index("predicted_infringement")
_AC_UNDER_CONTROL = _AC_FIELDS.index("under_control")
_GLOBAL_TIME, _GLOBAL_ACTIVE, _GLOBAL_MAX_INFR = 0, 1, 3
# AMAN-sequence columns (Config.SEQUENCE_OBS): appended after the 12 base features.
_AC_SEQ = len(_AC_FIELDS)  # first sequence column


def map_window_time(seconds: float) -> float:
    """Encode a non-negative time (s) on the window's scale, to [-1, 1] (signed-log, as the
    base time_to_target encoding, but over WindowedConfig.TIME_SCALE_S)."""
    return _map01_to_sym(
        map_signed_log(
            max(0.0, float(seconds)),
            float(WindowedConfig.TIME_SCALE_S),
            float(Config.TIME_TO_TARGET_LOG_FLOOR_S),
        )
    )


def landing_bonus(abs_dev_s: float) -> float:
    """Per-flight reward paid once, at landing, from the REALISED |deviation| (s)."""
    kappa = float(WindowedConfig.LANDING_BONUS_KAPPA_S)
    total = 0.0
    for threshold_s, weight in WindowedConfig.LANDING_BONUS_RUNGS:
        x = (float(threshold_s) - float(abs_dev_s)) / kappa
        total += float(weight) / (1.0 + math.exp(max(-50.0, min(50.0, -x))))
    return total


def landing_value(abs_dev_s: float, brackets=None) -> float:
    """Objective paid per flight at touchdown in REWARD_MODE "outcome_pbrs": a STEP function of
    the deviation bracket (WindowedConfig.LANDING_BRACKETS, or the env's "landing_brackets"), so
    where a flight lands inside its bracket does not change the score -- only the bracket, safety
    and the clearance count do."""
    d = abs(float(abs_dev_s))
    for limit, value in (brackets or WindowedConfig.LANDING_BRACKETS):
        if d <= float(limit):
            return float(value)
    return float(WindowedConfig.LANDING_VALUE_BEYOND)


def landing_value_smooth(abs_dev_s: float, brackets=None) -> float:
    """The bracket step function with each step replaced by a sigmoid of width
    LANDING_SMOOTH_KAPPA_S. Used only inside the potential Phi_dev, for gradient."""
    d = abs(float(abs_dev_s))
    k = float(WindowedConfig.LANDING_SMOOTH_KAPPA_S)
    brackets = list(brackets or WindowedConfig.LANDING_BRACKETS)
    value = float(WindowedConfig.LANDING_VALUE_BEYOND)
    below = [float(WindowedConfig.LANDING_VALUE_BEYOND)] + [float(v) for _, v in reversed(brackets)]
    # value(d) = beyond + sum over boundaries of (step height) * [d <= boundary], smoothed
    for (limit, _), lower, upper in zip(reversed(brackets), below[:-1], below[1:]):
        x = (float(limit) - d) / k
        value += (upper - lower) / (1.0 + math.exp(max(-50.0, min(50.0, -x))))
    return value


class WindowedATCEnv(ATCSingleAgentEnv):
    """ATCSingleAgentEnv over a 20-flight stream, exposing a sliding window of the next 10."""

    def __init__(self, config: Dict[str, Any] | None = None, verbose: bool = False):
        super().__init__(config, verbose)
        assert int(WindowedConfig.WINDOW_SIZE) == int(Config.N_AIRCRAFT_MAX), (
            "WINDOW_SIZE must equal Config.N_AIRCRAFT_MAX: the window IS the slot range"
        )
        assert float(WindowedConfig.DEVIATION_RAMP_S) <= float(WindowedConfig.ROLLOUT_LOOKAHEAD_S), (
            "every flight on the deviation ramp must be inside the rollout"
        )
        config = config or {}

        # The base __init__ built its simulator for Config's default difficulty. Rebuild it for
        # the windowed scenario so the experiment does not follow whatever config.py defaults to.
        # Config key "use_case": 1 = MXP trombone (WindowedConfig.DIFFICULTY), 2 = BGY point merge.
        use_case = config.get("use_case", None)
        self.difficulty = (getattr(MissionDifficulty, f"VALIDATION_USE_CASE_{int(use_case)}") if use_case is not None
                           else WindowedConfig.DIFFICULTY)
        if self.difficulty == MissionDifficulty.VALIDATION_USE_CASE_2 and str(Config.ACTION_SET) == "v2":
            print("[windowed] WARNING: point merge with the trombone clearance set v2 (routes cut, lengthen/"
                  "shorten the arc); use TADA_ACTION_SET=v2pms for the point-merge procedure.")
        simulation_config_path = config.get("simulation_config_path", None) or get_config_path(difficulty=self.difficulty)
        self.simulator = Simulator(
            self.difficulty, simulation_config_path, self._route_trombone_store, self._verbose
        )
        # Per-env override so an evaluation can run a longer stream than training.
        self.simulator.scenario_aircraft_count = int(
            config.get("scenario_aircraft_count", WindowedConfig.SCENARIO_AIRCRAFT_COUNT)
        )
        # Stitched streams (see WindowedConfig.STITCH_SEGMENTS).
        self.simulator.stitch_segments = int(config.get("stitch_segments", WindowedConfig.STITCH_SEGMENTS))
        self.simulator.stitch_gap_s = tuple(config.get("stitch_gap_s", WindowedConfig.STITCH_GAP_S))
        # Episode-length cap. A longer stream needs a higher one: 40 flights run to ~10000 s
        # (~225 steps). The prediction rollout is receding, so this only moves the end.
        self._max_episode_steps_cap = int(config.get(
            "max_episode_steps_cap",
            WindowedConfig.MAX_EPISODE_STEPS * max(1, self.simulator.stitch_segments),
        ))

        self._eta: Dict[str, float] = {}
        self._window: List[Aircraft] = []
        self._landed: Set[str] = set()
        self._landing_devs: Dict[str, float] = {}
        self._window_occupancy: List[int] = []
        self._window_overflow_steps = 0
        self._landing_bonus_total = 0.0
        self._phi_prev: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        # Per env (config key "reward_mode"), because SubprocVecEnv workers re-import config.
        self._reward_mode = str(config.get("reward_mode", WindowedConfig.REWARD_MODE))
        assert self._reward_mode in ("legacy", "outcome_pbrs"), self._reward_mode
        # The clearance cost is sized so a WHOLE stream's worst-case clearances stay below the
        # smallest bracket gap (WindowedConfig.CLEARANCE_COST, for one 20-flight segment). A
        # stitched stream of k segments is ~k times as long, so the per-clearance cost is divided
        # by k to keep the lexicographic ordering exact.
        self._clearance_cost = float(WindowedConfig.CLEARANCE_COST) / max(1, self.simulator.stitch_segments)
        # Reproducible training scenarios (config key "train_seed"): each training episode draws its
        # scenario seed from this env's own seeded generator (never one of the validation seeds),
        # and the scenario, stitching gaps and observation frame follow from it. Unset: the old
        # unseeded behaviour. Evaluation and pinned scenarios are unaffected.
        ts = config.get("train_seed", None)
        self._train_rng = np.random.default_rng(int(ts)) if ts is not None else None
        self._holdout_seeds = {int(x) for x in Config.SCENARIO_EVAL_SEEDS}
        # outcome_pbrs potentials without the time ramps (config key "pbrs_flat_potentials"): every
        # predicted AMAN-order inversion and every predicted conflict counts in full however far out.
        # Still a function of the state only, so the shaping still cannot change the optimal policy.
        self._pbrs_flat = bool(config.get("pbrs_flat_potentials", False))
        # outcome_pbrs objective ladder; per env (config key "landing_brackets") so a run can target
        # +/-30 s (WindowedConfig.HARD_BRACKETS). Ascending limits, values strictly decreasing.
        self._landing_brackets = tuple((float(a), float(b)) for a, b in
                                       config.get("landing_brackets", WindowedConfig.LANDING_BRACKETS))
        limits = [a for a, _ in self._landing_brackets]
        values = [b for _, b in self._landing_brackets] + [float(WindowedConfig.LANDING_VALUE_BEYOND)]
        assert limits == sorted(limits) and all(x > y for x, y in zip(values, values[1:])), self._landing_brackets
        # outcome_pbrs bust penalty; per env (config key) so a run can change it: workers re-import config.
        # Default: WindowedConfig's rule applied to this env's ladder (BASE + (best - beyond) x WINDOW_SIZE;
        # 90 on the default ladder, 100 on HARD_BRACKETS), so safety still outranks every bracket.
        default_penalty = float(Config.REWARD_VIOLATION_PENALTY) + (
            values[0] - float(WindowedConfig.LANDING_VALUE_BEYOND)) * int(WindowedConfig.WINDOW_SIZE)
        self._pbrs_violation_penalty = float(config.get("pbrs_violation_penalty", default_penalty))

        # Release (config key "release_locked"): the front of the landing queue is handed off while
        # each flight is locked -- under control, its predicted landing inside the prediction and
        # within "release_dev_s" of its AMAN target, and in no predicted conflict. A released flight
        # leaves the window (the next flight in the queue enters it) and can no longer be cleared;
        # the flights behind it still see it through their gap and conflict features.
        self._release_locked = bool(config.get("release_locked", False))
        self._release_dev_s = float(config.get("release_dev_s", 30.0))
        # "pay_at_release": a released flight's bracket value is paid at release (from its predicted
        # deviation) and it leaves the deviation potential; at touchdown only the difference to the
        # realised bracket is paid, so each flight still earns exactly landing_value(realised).
        self._pay_at_release = bool(config.get("pay_at_release", False))
        assert not self._pay_at_release or (self._release_locked and self._reward_mode == "outcome_pbrs"), \
            "pay_at_release needs release_locked and reward_mode outcome_pbrs"
        # "pbrs_flat_deviation": the deviation potential weighs every flight whose landing the
        # prediction resolves in full, instead of ramping in over the last DEVIATION_RAMP_S, so a
        # correction made early is credited as soon as it changes the prediction.
        self._flat_dev = bool(config.get("pbrs_flat_deviation", False))
        # "train_max_early_s": training scenarios are redrawn until no flight is predicted (do-nothing,
        # at t = 0) more than this early -- feasible streams only (650 s is the generator's own cap on
        # the delay a flight may need). Evaluation and pinned scenarios are never screened.
        tme = config.get("train_max_early_s", None)
        self._train_max_early_s = float(tme) if tme is not None else None
        self._train_redraws = 0
        # "train_seed_pool": a file of pre-screened scenario seeds (analysis/long_stream_seeds.py), one per
        # line; each training reset draws one with the env's train_seed generator. Redrawing at reset
        # (train_max_early_s) stalls the whole vectorised batch while one env simulates candidate after
        # candidate; with ~11% of stitched 2x20 streams feasible that tripled the step time.
        pool = config.get("train_seed_pool", None)
        self._train_pool = [int(x) for x in open(pool).read().split()] if pool else None
        if self._train_pool:
            self._train_pool = [x for x in self._train_pool if x not in {int(v) for v in Config.SCENARIO_EVAL_SEEDS}]
        self._released: Dict[str, float] = {}      # callsign -> predicted deviation at release
        self._release_paid: Dict[str, float] = {}  # callsign -> bracket value paid at release
        self._release_correction = 0.0             # sum of |touchdown corrections| (0 if predictions hold)

        # Reselection (see the module docstring). The spaces change only when it is on.
        self._multi_pick = int(Config.MAX_PICKS) > 1
        self._max_picks = int(config.get("max_picks", Config.MAX_PICKS))
        self._pending_cmds: List[Any] = []  # commands queued at this step, applied at its advance
        self._picked: List[str] = []         # callsigns cleared at this step, in order
        self._again_allowed = False
        self._again_picks = 0                # zero-time picks this episode
        self._multi_pick_steps = 0           # clock steps that issued two or more clearances
        if self._multi_pick:
            n_air = int(Config.N_AIRCRAFT_MAX)
            self.action_space = spaces.MultiDiscrete([n_air, int(Config.N_CLEARANCES), 2])
            self.observation_space = spaces.Dict({
                **self.observation_space.spaces,
                "mask_select": spaces.MultiBinary(n_air),
                "mask_again": spaces.MultiBinary(1),
            })

    # ---------------------- window ----------------------

    def _compute_window(self) -> Tuple[List[Aircraft], int]:
        """Next WINDOW_SIZE flights in the landing queue, and how many under-control flights
        were left OUT of it (aircraft the agent is responsible for but cannot see)."""
        assert self.simulator.world is not None
        unlanded = [ac for ac in self.simulator.world.aircraft
                    if not ac.mission.completed and ac.call_sign not in self._released]
        queue = (
            [ac for ac in unlanded if ac.under_control]
            if WindowedConfig.WINDOW_UNDER_CONTROL_ONLY
            else list(unlanded)
        )
        if not queue:
            # Only reachable with WINDOW_UNDER_CONTROL_ONLY, in a gap between arrivals: mirror
            # get_valid_aircraft_list() so the policy never sees an empty slate mid-stream.
            queue = unlanded
        if not queue and self._released:
            # Every flight still airborne is released (the stream's tail): show those rather than an
            # empty slate; they stay DO_NOTHING-only (_build_window_observation masks them).
            queue = [ac for ac in self.simulator.world.aircraft
                     if not ac.mission.completed and ac.call_sign in self._released]
        queue.sort(key=lambda ac: self._eta.get(ac.call_sign, math.inf))
        window = queue[: int(WindowedConfig.WINDOW_SIZE)]
        in_window = {ac.call_sign for ac in window}
        hidden_active = sum(1 for ac in unlanded if ac.under_control and ac.call_sign not in in_window)
        return window, hidden_active

    def _sequence_state(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Per flight: (rank shift, crossings, gap error to AMAN predecessor, to successor).

        Landing times are PREDICTED for flights still airborne (from the do-nothing rollout: the
        MISS_TIMED_LANDING event, or the AMAN target when the rollout lands the flight within the
        simulator's 10 s tolerance; a flight the rollout does not land is placed at the rollout's
        end, i.e. "at least that late") and ACTUAL for flights already down. The AMAN sequence is
        fixed; these describe how far the predicted traffic departs from it:

          rank shift  predicted landing rank - AMAN rank (+ = being overtaken, - = overtaking)
          crossings   flights it is predicted to cross, either direction
          gap errors  (predicted landing gap to the AMAN predecessor / successor) - (target gap);
                      negative = compression, which precedes both conflicts and misses
        """
        land_t = self._predicted_landing_times()
        if not land_t:
            return {}
        aman = sorted(self._eta, key=self._eta.get)
        pred_order = sorted(aman, key=lambda c: (land_t[c], self._eta[c]))
        pred_rank = {c: i for i, c in enumerate(pred_order)}
        out = {}
        for i, cs in enumerate(aman):
            crossings = sum(
                1 for o in aman
                if o != cs and (self._eta[o] < self._eta[cs]) != (land_t[o] < land_t[cs])
            )
            gp = gs = 0.0
            if i > 0:
                p_ = aman[i - 1]
                gp = (land_t[cs] - land_t[p_]) - (self._eta[cs] - self._eta[p_])
            if i + 1 < len(aman):
                n_ = aman[i + 1]
                gs = (land_t[n_] - land_t[cs]) - (self._eta[n_] - self._eta[cs])
            out[cs] = (float(pred_rank[cs] - i), float(crossings), float(gp), float(gs))
        return out

    def _predicted_landing_times(self) -> Dict[str, float]:
        """callsign -> landing time (absolute s): ACTUAL for flights already down, PREDICTED by
        the do-nothing rollout otherwise -- its MISS_TIMED_LANDING event, or the AMAN target when
        the rollout lands the flight within the simulator's 10 s tolerance, or the rollout's end
        ("at least that late") when the rollout does not land it."""
        return self._predicted_landings()[0]

    def _predicted_landings(self) -> Tuple[Dict[str, float], Set[str]]:
        """(landing times as above, the airborne flights whose landing the rollout resolves)."""
        from flight_simulator import Infringement

        pred = self.current_predicted_world
        if pred is None or not self._eta:
            return {}, set()
        rollout_end = float(pred.age_in_seconds)
        event_t = {}
        for infr in pred.infringements:
            if isinstance(infr, Infringement.MISS_TIMED_LANDING):
                event_t[str(infr.aircraft)] = float(infr.timestamp_s)
        landed_in_rollout = {ac.call_sign for ac in pred.aircraft if ac.mission.completed}
        land_t = {}
        for cs, eta in self._eta.items():
            if cs in self._landing_devs:                       # actually landed
                land_t[cs] = eta + float(self._landing_devs[cs])
            elif cs in event_t:
                land_t[cs] = event_t[cs]
            elif cs in landed_in_rollout:
                land_t[cs] = eta                               # within the 10 s tolerance
            else:
                land_t[cs] = max(eta, rollout_end)             # beyond the prediction
        resolved = (set(event_t) | landed_in_rollout) - set(self._landing_devs)
        return land_t, resolved

    def _potential(self) -> Tuple[float, float, float, float]:
        """Phi(s) = (dev, conf, seq, gap) for REWARD_MODE "outcome_pbrs" (see WindowedConfig).

        All four come from the do-nothing prediction, a deterministic function of the state, so
        the shaping gamma * Phi(s') - Phi(s) is potential-based and cannot change which policy is
        optimal. Proximity weight w = 1 - min(1, time-to-landing / DEVIATION_RAMP_S).
        """
        assert self.simulator.world is not None
        land_t, resolved = self._predicted_landings()
        if not land_t:
            return (0.0, 0.0, 0.0, 0.0)
        now = float(self.simulator.world.age_in_seconds)
        ramp = float(WindowedConfig.DEVIATION_RAMP_S)

        def w(t_land: float) -> float:
            return 1.0 - min(1.0, max(0.0, t_land - now) / ramp)

        unlanded = [c for c in sorted(self._eta, key=self._eta.get) if c not in self._landing_devs]

        # Predicted value of the objective still to be collected (a flight paid at release has
        # none left). Flat: full weight for every flight whose landing the prediction resolves.
        w_dev = (lambda c: 1.0 if c in resolved else 0.0) if self._flat_dev else (lambda c: w(land_t[c]))
        phi_dev = sum(w_dev(c) * landing_value_smooth(abs(land_t[c] - self._eta[c]), self._landing_brackets)
                      for c in unlanded if c not in self._release_paid)

        # Predicted conflicts touching the window (the predicted conflict term, as a potential).
        phi_conf = float(infringement_reward_from_world(
            self.current_predicted_world, now,
            weight=float(WindowedConfig.PHI_CONFLICT_WEIGHT_FLAT if self._pbrs_flat else WindowedConfig.PHI_CONFLICT_WEIGHT),
            callsigns=self._window_callsigns(), flat=self._pbrs_flat,
        ))

        # Predicted AMAN-order inversions among unlanded flights, weighted by the earlier landing.
        phi_seq = 0.0
        for i, a in enumerate(unlanded):
            for b in unlanded[i + 1:]:
                if land_t[a] > land_t[b]:
                    phi_seq -= 1.0 if self._pbrs_flat else w(min(land_t[a], land_t[b]))
        phi_seq *= float(WindowedConfig.PHI_SEQ_PER_SWAP)

        # Compression below the minimum landing spacing, AMAN-consecutive pairs whose later
        # flight is still airborne (gap clipped at 0: an inversion is priced by phi_seq).
        phi_gap = 0.0
        aman = sorted(self._eta, key=self._eta.get)
        g_min = float(WindowedConfig.PHI_GAP_MIN_S)
        for prev_, next_ in zip(aman, aman[1:]):
            if next_ in self._landing_devs:
                continue
            gap = max(0.0, land_t[next_] - land_t[prev_])
            if gap < g_min:
                phi_gap -= w(land_t[next_]) * (g_min - gap) / 60.0
        phi_gap *= float(WindowedConfig.PHI_GAP_PER_MIN)
        return (float(phi_dev), float(phi_conf), float(phi_seq), float(phi_gap))

    def _pbrs(self) -> bool:
        return self._reward_mode == "outcome_pbrs"

    def _window_callsigns(self) -> Set[str]:
        return {ac.call_sign for ac in self._window}

    def _unlanded_callsigns(self) -> Set[str]:
        assert self.simulator.world is not None
        return {ac.call_sign for ac in self.simulator.world.aircraft if not ac.mission.completed}

    def _rollout_steps(self) -> int:
        """Receding prediction horizon: the fixed lookahead or the window's last scheduled
        landing, whichever is further, plus a margin for late flights. Bounded by the window,
        so it is the same at every point of an arbitrarily long stream."""
        assert self.simulator.world is not None
        tba = float(Config.TIME_BETWEEN_ACTIONS)
        now_s = float(self.simulator.world.age_in_seconds)
        window_span_s = max(
            (self._eta.get(ac.call_sign, now_s) - now_s for ac in self._window), default=0.0
        )
        span_s = max(float(WindowedConfig.ROLLOUT_LOOKAHEAD_S), window_span_s)
        steps = math.ceil(span_s / tba) + int(WindowedConfig.ROLLOUT_MARGIN_STEPS)
        return int(min(steps, int(WindowedConfig.ROLLOUT_MAX_STEPS)))

    def _predict(self, world, commands=None):
        """The do-nothing prediction; with ``commands`` (reselection's queued clearances), the
        same rollout with them applied: what the clock step will realise if nothing else is
        issued."""
        if not commands:
            return self.simulator.advance_multiple_steps(steps=self._rollout_steps(), world=world)
        return simulate_n_next_states(
            self._rollout_steps() * int(Config.TIME_BETWEEN_ACTIONS),
            1000,
            world,
            TimedCommandList(commands=list(commands)).to_tuple_list(),
        )

    def _with_queued(self, action_cmds):
        """This pick's commands with the step's queued ones in front (reselection)."""
        if not self._pending_cmds:
            return action_cmds
        return TimedCommandList(commands=list(self._pending_cmds) + list(action_cmds.commands))

    def _build_window_observation(self, count: bool = True):
        """count=False for a zero-time pick: the window statistics are per clock step."""
        self._window, hidden_active = self._compute_window()
        if count:
            self._window_occupancy.append(len(self._window))
            if hidden_active > 0:
                self._window_overflow_steps += 1

        obs, order = build_global_observation(
            simulator=self.simulator,
            curr_world=self.current_predicted_world,
            global_action_history=self._action_history,
            perm=self.acft_perm,
            route_trombone_store=self._route_trombone_store,
            T_obs=self.obs_matrix,
            aircraft_list=self._window,
        )

        # --- Observables that encoded the SCENARIO rather than the window, replaced ---
        now_s = float(self.simulator.world.age_in_seconds)
        slots = [i for i, cs in enumerate(order) if cs]

        # Per-aircraft time_to_target: the base builder reads it from the END of the prediction
        # rollout, which yields (rollout length - time to go) -- a function of rollout length,
        # and so of scenario size, that also sits pinned at the top of the 2880 s scale. Use the
        # scheduled time to go instead (AMAN target minus now), on the window's time scale.
        for i in slots:
            obs["aircraft"][i, _AC_TTT] = np.float32(
                map_window_time(self._eta.get(order[i], now_s) - now_s)
            )

        # Global 0, time: absolute scenario time saturates after 2880 s and means nothing in a
        # stream. Scheduled time until the window's LAST flight is due: the span of traffic the
        # agent is looking at, the same quantity at any point of any stream.
        last_eta = max((self._eta.get(order[i], now_s) for i in slots), default=now_s)
        obs["global"][_GLOBAL_TIME] = np.float32(map_window_time(last_eta - now_s))

        # Global 1, active aircraft: under a queue window the slot count is ~always WINDOW_SIZE
        # and only drops as the finite scenario runs dry -- an end-of-scenario signal. Count the
        # window flights actually under control: the current traffic load.
        n_active = sum(1 for i in slots if obs["aircraft"][i, _AC_UNDER_CONTROL] > 0.5)
        obs["global"][_GLOBAL_ACTIVE] = np.float32(
            map_to_sym_unit(float(n_active), 0.0, float(WindowedConfig.WINDOW_SIZE))
        )

        # Global 3, max predicted infringement: was taken over every aircraft in the world.
        # Scope it to the window (same encoding as the per-aircraft column it is a max of). A
        # window flight's own column still includes conflicts with a partner outside the window.
        if slots:
            obs["global"][_GLOBAL_MAX_INFR] = np.float32(max(obs["aircraft"][i, _AC_PIN] for i in slots))
        else:
            obs["global"][_GLOBAL_MAX_INFR] = np.float32(map_to_sym_unit(0.0, 0.0, 1.0))

        # AMAN-sequence columns (Config.SEQUENCE_OBS). Encodings: rank shift and crossings are
        # counts, clipped at 10 and mapped linearly; the gap errors use the same signed-log
        # encoding as the schedule deviation, whose resolution sits at the +/-60 s scale.
        if Config.SEQUENCE_OBS and slots:
            seq = self._sequence_state()
            for i in slots:
                rank, cross, gp, gs = seq.get(order[i], (0.0, 0.0, 0.0, 0.0))
                obs["aircraft"][i, _AC_SEQ + 0] = np.float32(np.clip(rank / 10.0, -1.0, 1.0))
                obs["aircraft"][i, _AC_SEQ + 1] = np.float32(map_to_sym_unit(min(cross, 10.0), 0.0, 10.0))
                obs["aircraft"][i, _AC_SEQ + 2] = np.float32(map_deviation(gp))
                obs["aircraft"][i, _AC_SEQ + 3] = np.float32(map_deviation(gs))

        # Flights that have not spawned hold their place in the queue but can only be left alone;
        # so can released flights (shown only when nothing else is airborne).
        for i in slots:
            pending = not WindowedConfig.PENDING_ACTIONABLE and obs["aircraft"][i, _AC_UNDER_CONTROL] < 0.5
            if pending or order[i] in self._released:
                obs["mask_action_per_ac"][i, :] = 0
                obs["mask_action_per_ac"][i, int(Action.DO_NOTHING)] = 1

        if self._multi_pick:
            self._add_pick_observation(obs, order, slots)

        self._obs = obs
        return obs, order

    def _add_pick_observation(self, obs, order, slots) -> None:
        """Reselection: the "cleared at this step" column and the two masks.

        On a step's first pick nothing is cleared yet: the column is all zero and mask_select is
        exactly the policy's old rule (under-control slots, else every visible slot).
        """
        picked = set(self._picked)
        col = int(Config.AIRCRAFT_SCALAR_LEN) - int(Config.PICK_FEATURES)
        under = [i for i in slots if obs["aircraft"][i, _AC_UNDER_CONTROL] > 0.5]
        free = [i for i in under if order[i] not in picked]
        for i in slots:
            if order[i] in picked:
                # One clearance per aircraft per step: it can only be left alone now.
                obs["aircraft"][i, col] = np.float32(1.0)
                obs["mask_action_per_ac"][i, :] = 0
                obs["mask_action_per_ac"][i, int(Action.DO_NOTHING)] = 1
        mask_select = np.zeros((int(Config.N_AIRCRAFT_MAX),), dtype=np.int8)
        # free is empty only if every under-control aircraft is cleared, which mask_again rules
        # out; fall back to the old rule rather than an all-masked row.
        for i in (free or under or slots):
            mask_select[i] = 1
        # Another pick after this one needs budget and an aircraft left once this one is taken.
        self._again_allowed = len(self._picked) + 1 < self._max_picks and len(free) >= 2
        obs["mask_select"] = mask_select
        obs["mask_again"] = np.array([int(self._again_allowed)], dtype=np.int8)

    def _prediction_horizon_steps(self) -> int:
        return self._rollout_steps()

    def _side_view_highlight(self):
        return self._window_callsigns()

    def _render_title(self) -> str:
        n = len(self._eta)
        on_time = sum(1 for d in self._landing_devs.values() if abs(d) <= float(WindowedConfig.ON_TIME_DEV_S))
        # Short: the title is one large centred line and clips beyond ~45 characters. Window
        # given as queue positions (1-based), on-time as a fraction of flights landed so far.
        queue = sorted(self._eta, key=self._eta.get)
        pos = sorted(queue.index(ac.call_sign) + 1 for ac in self._window)
        span = f"{pos[0]}-{pos[-1]}" if pos else "-"
        return f"seed {self.scenario_seed}  window {span}/{n}  on time {on_time}/{len(self._landing_devs)}"

    # ---------------------- snapshot / restore (inference-time lookahead) ----------------------

    def snapshot(self) -> Dict[str, Any]:
        """Everything step() can change, so a candidate action can be tried and undone exactly.

        Simulator worlds are immutable on the Python side (every advance returns a new World),
        so references suffice. Containers that step() mutates IN PLACE are copied, the trombone
        route store (mutated by trombone clearances) is cloned, and the simulator's own
        attributes are copied shallowly. Verified by replaying an episode through
        snapshot/restore with the policy's own choice: identical to a plain run.
        """
        import copy as _copy

        snap = dict(self.__dict__)
        snap["__simulator__"] = dict(self.simulator.__dict__)
        snap["__store_parts__"] = {k: v.clone() for k, v in self._route_trombone_store._route_parts.items()}
        snap["__mutable__"] = {
            "_action_history": list(self._action_history or []),
            "_past_tracks": {cs: list(t) for cs, t in (self._past_tracks or {}).items()},
            "_landed": set(self._landed),
            "_landing_devs": dict(self._landing_devs),
            "_action_counts": dict(self._action_counts or {}),
            "_window_occupancy": list(self._window_occupancy),
            "_window": list(self._window),
            "_agent_colors": dict(self._agent_colors or {}),
            "last_reward": _copy.copy(self.last_reward),
            "_pending_cmds": list(self._pending_cmds),
            "_picked": list(self._picked),
            "_released": dict(self._released),
            "_release_paid": dict(self._release_paid),
        }
        return snap

    def restore(self, snap: Dict[str, Any]) -> None:
        """Return to a snapshot. Copies again, so one snapshot can be restored many times."""
        import copy as _copy

        base = {k: v for k, v in snap.items() if not k.startswith("__")}
        self.__dict__.clear()
        self.__dict__.update(base)
        m = snap["__mutable__"]
        self._action_history = list(m["_action_history"])
        self._past_tracks = {cs: list(t) for cs, t in m["_past_tracks"].items()}
        self._landed = set(m["_landed"])
        self._landing_devs = dict(m["_landing_devs"])
        self._action_counts = dict(m["_action_counts"])
        self._window_occupancy = list(m["_window_occupancy"])
        self._window = list(m["_window"])
        self._agent_colors = dict(m["_agent_colors"])
        self.last_reward = _copy.copy(m["last_reward"])
        self._pending_cmds = list(m["_pending_cmds"])
        self._picked = list(m["_picked"])
        self._released = dict(m["_released"])
        self._release_paid = dict(m["_release_paid"])
        self.simulator.__dict__.clear()
        self.simulator.__dict__.update(snap["__simulator__"])
        # The store object is shared with the simulator: refill it in place, from fresh clones.
        self._route_trombone_store._route_parts = {k: v.clone() for k, v in snap["__store_parts__"].items()}

    # ---------------------- gym API ----------------------

    def reset(self, *, seed=None, options=None):
        # The base reset needs the ETAs cleared first: it builds a (discarded) default obs.
        self._eta = {}
        self._pending_cmds, self._picked = [], []
        self._again_picks = self._multi_pick_steps = 0
        options = dict(options or {})
        draw_seed = self._train_rng is not None and not self.evaluation_mode and options.get("scenario_seed") is None
        screen = self._train_max_early_s is not None and not self.evaluation_mode and options.get("scenario_seed") is None
        for _ in range(200):
            if draw_seed and self._train_pool:
                s = int(self._train_pool[int(self._train_rng.integers(len(self._train_pool)))])
                options["scenario_seed"] = s
                np.random.seed(s % 2**32)
            elif draw_seed:
                s = int(self._train_rng.integers(0, 2**31 - 1))
                while s in self._holdout_seeds:
                    s = int(self._train_rng.integers(0, 2**31 - 1))
                options["scenario_seed"] = s
                np.random.seed(s % 2**32)  # the observation frame, as the evaluation scripts pin it
            super().reset(seed=seed, options=options)
            assert self.simulator.world is not None
            if not screen or self._max_early_s() <= self._train_max_early_s:
                break
            self._train_redraws += 1  # over capacity: draw another training scenario

        self._eta = {
            str(t.ac_callsign): float(t.eta_s) for t in self.simulator.world.arrival_targets
        }
        self._landed = {
            ac.call_sign for ac in self.simulator.world.aircraft if ac.mission.completed
        }
        self._landing_devs = {}
        self._released, self._release_paid, self._release_correction = {}, {}, 0.0
        self._window_occupancy = []
        self._window_overflow_steps = 0
        self._landing_bonus_total = 0.0

        # Render-only: the base env colours by tab10, which repeats every 10 aircraft. Colour
        # by queue position on tab20 so a 20-flight stream has 20 distinct colours.
        cmap = plt.get_cmap("tab20")
        queue = sorted(self._eta, key=self._eta.get)
        self._agent_colors = {cs: to_rgba(cmap(i % 20)) for i, cs in enumerate(queue)}

        # The base reset predicted to the end of the scenario; replace it with the receding
        # horizon (the window has to exist first, since the horizon covers it).
        self._window, _ = self._compute_window()
        self.current_predicted_world = self._predict(self.simulator.world)

        obs, self._prev_aircraft_order = self._build_window_observation()
        assert self._prev_aircraft_order is not None and len(self._prev_aircraft_order) > 0
        self._phi_prev = self._potential() if self._pbrs() else (0.0, 0.0, 0.0, 0.0)
        return obs, {}

    def step(self, action: Tuple[int, int] | np.ndarray):  # type: ignore[override]
        assert self.simulator.world is not None, "World is not initialized"
        n_clearances = int(Config.N_CLEARANCES)

        again = 0
        action_arr = np.asarray(action).ravel()
        a_air, a_clr = int(action_arr[0]), int(action_arr[1])
        again = int(action_arr[2]) if action_arr.size > 2 else 0

        callsign = self._prev_aircraft_order[a_air]
        do_nothing = int(Action.DO_NOTHING)
        if self._multi_pick:
            # The masks already say this; enforced here too, for callers that bypass them
            # (replays, shields). A (aircraft, clearance) action is again=0.
            if callsign in self._picked:
                a_clr = do_nothing
            again = int(bool(again) and self._again_allowed and a_clr != do_nothing)
        else:
            again = 0
        self.action = a_air * n_clearances + a_clr

        commanded_ac = self.simulator.world.get_aircraft_by_callsign(callsign)
        remaining_wps = len(commanded_ac.route) if (commanded_ac and commanded_ac.route) else 0

        action_cmds = interpret_action(a_clr, callsign, self.simulator, self._route_trombone_store)
        if again:
            return self._queue_pick(action_cmds, callsign, a_air, a_clr, remaining_wps)

        decision_time_s = float(self.simulator.world.age_in_seconds)
        n_picks = len(self._picked) + int(a_clr != do_nothing)
        self._advance_and_compute_reward(action_cmds, callsign, a_clr, remaining_wps=remaining_wps)
        self._pending_cmds, self._picked = [], []
        self._multi_pick_steps += int(n_picks >= 2)

        if has_near_conflict(self.current_predicted_world, float(self.simulator.world.age_in_seconds)):
            self._near_conflict_violated = True
            self._steps_since_near_conflict = 0
        else:
            self._steps_since_near_conflict += 1

        # --- Per-flight landing reward: every flight that touched down during this step ---
        realised = get_aircraft_time_deviations(self.simulator.world)
        bonus = 0.0
        for ac in self.simulator.world.aircraft:
            if ac.mission.completed and ac.call_sign not in self._landed:
                self._landed.add(ac.call_sign)
                dev = float(realised.get(ac.call_sign, 0.0))
                self._landing_devs[ac.call_sign] = dev
                if self._pbrs():
                    value = landing_value(abs(dev), self._landing_brackets)
                    paid = self._release_paid.get(ac.call_sign)
                    if paid is not None:  # paid at release: only the difference to the realised bracket
                        self._release_correction += abs(value - paid)
                        value -= paid
                    bonus += value
                else:
                    bonus += landing_bonus(abs(dev))
        self._landing_bonus_total += bonus

        terminated = self._check_terminated()
        truncated = self._check_truncated(terminated)
        busted = self.last_end_reason in ("separation", "airspace_exit")
        if WindowedConfig.CONTINUING_TASK and terminated and not busted:
            # The stream ran dry. In a continuing task more flights would follow, so bootstrap.
            terminated, truncated = False, True

        if self._release_locked and not busted:
            paid = self._release_front()
            bonus += paid
            self._landing_bonus_total += paid

        self._terminal_tier = 0
        if busted:
            bonus -= self._violation_penalty()
            # A release payment is an advance on the touchdown bracket: a flight that never lands
            # (the episode ends in a bust) gives it back, so a bust costs what it would without
            # pay_at_release and safety keeps its lexicographic priority.
            clawback = sum(v for c, v in self._release_paid.items() if c not in self._landing_devs)
            bonus -= clawback
            self._landing_bonus_total -= clawback
        elif terminated or truncated:
            # Reporting only: the tier over ALL flights of the scenario, as in the base env.
            _, tier, _ = self._evaluate_success(self.current_predicted_world)
            self._terminal_tier = int(tier)

        self.last_reward.bonus += bonus
        self.cumulative_reward += bonus

        episode_metrics = self._build_episode_metrics() if (terminated or truncated) else None

        obs, self._prev_aircraft_order = self._build_window_observation()
        assert self._prev_aircraft_order is not None and len(self._prev_aircraft_order) > 0

        if self._pbrs():
            # Potential-based shaping, per component (logged separately). Phi(s') is taken after
            # the new window is built; a bust is absorbing, so Phi(s') = 0 there. Truncation keeps
            # the real Phi(s'): PPO bootstraps, and a stream that ran dry has Phi ~ 0 anyway.
            gamma = float(Config.PBRS_GAMMA)
            phi_new = (0.0, 0.0, 0.0, 0.0) if busted else self._potential()
            f = [gamma * n - o for n, o in zip(phi_new, self._phi_prev)]
            self._phi_prev = phi_new
            self.last_reward.deviation += f[0]
            self.last_reward.conflict += f[1]
            self.last_reward.goal += f[2] + f[3]
            self.cumulative_reward += sum(f)

        r_parts = self.last_reward
        reward = float(r_parts.total())

        info = {
            "action": int(self.action),
            "reward": float(reward),
            "reward_parts": r_parts,
            "callsign": callsign,
            "aircraft_index": int(a_air),
            "clearance": int(a_clr),
            "end_reason": self.last_end_reason,
        }
        if self._multi_pick:
            info.update({"zero_time": False, "again": 0, "pick": int(n_picks), "time_s": decision_time_s})
        if episode_metrics is not None:
            info["episode_metrics"] = episode_metrics
            info["is_success"] = bool(episode_metrics["success"])
        return obs, reward, terminated, truncated, info

    def _max_early_s(self) -> float:
        """How early the earliest flight of the current scenario lands if nothing is done: the
        do-nothing rollout run to the stream's end (analysis/long_stream_seeds.py's screen)."""
        from actions.rewards import get_aircraft_time_deviations

        world = self.simulator.world
        last_eta = max(float(t.eta_s) for t in world.arrival_targets)
        steps = math.ceil(last_eta / float(Config.TIME_BETWEEN_ACTIONS)) + 40
        devs = get_aircraft_time_deviations(self.simulator.advance_multiple_steps(steps=steps, world=world))
        return max(0.0, -min(devs.values())) if devs else 0.0

    def _release_front(self) -> float:
        """Hand off the front of the landing queue while each flight is locked (see __init__):
        under control, its landing resolved by the do-nothing prediction within release_dev_s of
        its target, and in no predicted conflict. Returns the bracket value paid at release
        (0 unless pay_at_release)."""
        from flight_simulator import Infringement

        pred = self.current_predicted_world
        land_t, resolved = self._predicted_landings()
        in_conflict = set()
        for infr in (pred.infringements if pred is not None else []):
            if isinstance(infr, Infringement.LOSS_OF_SEPARATION):
                in_conflict.update(str(x) for x in infr.aircraft_pair)
        ac_by_cs = {ac.call_sign: ac for ac in self.simulator.world.aircraft}
        paid = 0.0
        for cs in sorted(self._eta, key=self._eta.get):
            if cs in self._landing_devs or cs in self._released:
                continue
            ac = ac_by_cs.get(cs)
            dev = land_t.get(cs, math.inf) - self._eta[cs]
            if (ac is None or not ac.under_control or cs not in resolved or cs in in_conflict
                    or abs(dev) > self._release_dev_s):
                break  # the queue's front is not locked: nothing behind it is released either
            self._released[cs] = float(dev)
            if self._pay_at_release:
                value = landing_value(abs(dev), self._landing_brackets)
                self._release_paid[cs] = value
                paid += value
        return paid

    def _queue_pick(self, action_cmds, callsign: str, a_air: int, a_clr: int, remaining_wps: int):
        """again=1: queue this clearance, keep the clock, and return a fresh observation.

        Nothing moves, so nothing can land, bust or run out of time here. The reward is this
        clearance's cost plus the change in the potential, which now includes the queued
        clearances in its prediction. No time passes, so the shaping is Phi(s') - Phi(s), with
        discount 1, and info["zero_time"] tells the learner to discount the transition by 1.
        """
        assert self.simulator.world is not None
        self._pending_cmds.extend(c for c in action_cmds.commands if c.command is not None)
        self._picked.append(callsign)
        self._again_picks += 1
        self.current_predicted_world = self._predict(self.simulator.world, self._pending_cmds)

        if self._pbrs():
            r_parts = RewardParts(deviation=0.0, conflict=0.0, action=-self._clearance_cost, goal=0.0)
        else:
            # Legacy (not for training with reselection): only the action penalty; the dense
            # terms are charged once per clock step.
            r_parts = RewardParts(deviation=0.0, conflict=0.0,
                                  action=-float(_action_penalty(action_cmds, remaining_wps)), goal=0.0)
        self._record_pick(r_parts, callsign, a_clr)

        obs, self._prev_aircraft_order = self._build_window_observation(count=False)

        if self._pbrs():
            phi_new = self._potential()
            f = [n - o for n, o in zip(phi_new, self._phi_prev)]
            self._phi_prev = phi_new
            r_parts.deviation += f[0]
            r_parts.conflict += f[1]
            r_parts.goal += f[2] + f[3]
            self.cumulative_reward += sum(f)

        reward = float(r_parts.total())
        info = {
            "action": int(self.action),
            "reward": reward,
            "reward_parts": r_parts,
            "callsign": callsign,
            "aircraft_index": int(a_air),
            "clearance": int(a_clr),
            "end_reason": self.last_end_reason,
            "zero_time": True,
            "again": 1,
            "pick": len(self._picked),
            "time_s": float(self.simulator.world.age_in_seconds),
        }
        return obs, reward, False, False, info

    def _record_pick(self, reward_parts: RewardParts, actor_id: str, action_num: int) -> None:
        """_update_render_state_after_advance for a zero-time pick: the same reward, tally and
        action-history bookkeeping (the history is what the policy's GRU reads), no tracks."""
        assert self.simulator.world is not None
        self.last_reward = reward_parts
        self.cumulative_reward += float(reward_parts.total())
        self._last_actor_id = actor_id
        self._last_action_num = int(action_num)
        if int(action_num) != int(Action.DO_NOTHING) and self._action_counts is not None:
            self._action_counts[actor_id] = self._action_counts.get(actor_id, 0) + 1
        ac = self.simulator.world.get_aircraft_by_callsign(actor_id)
        if ac is not None:
            pos = ac.state.position_nm_nm_ft
            self._action_history.append({
                "call_sign": actor_id,
                "action": int(action_num),
                "x": float(pos.x),
                "y": float(pos.y),
                "step": int(self._steps),
                "reward": float(reward_parts.total()),
                "reward_parts": reward_parts,
            })

    # ---------------------- overridden helpers ----------------------

    def _compute_episode_horizon(self) -> None:
        """Episode length (truncation only; the rollout has its own receding horizon)."""
        assert self.simulator.world is not None
        tba = float(Config.TIME_BETWEEN_ACTIONS)
        max_eta = max(
            (float(t.eta_s) for t in self.simulator.world.arrival_targets),
            default=Config.EPISODE_STEPS * tba,
        )
        self.episode_time_horizon = max_eta
        if self._max_steps_override is not None:
            self.max_steps_per_episode = int(self._max_steps_override)
            return
        steps = math.ceil(max_eta / tba) + int(WindowedConfig.HORIZON_STEP_MARGIN)
        self.max_steps_per_episode = int(
            min(max(steps, int(WindowedConfig.MIN_EPISODE_STEPS)), self._max_episode_steps_cap)
        )

    def _reward_scopes(self) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """(deviation callsigns, conflict callsigns) for the current state; None = no filter."""
        dev_callsigns = {
            "unlanded": self._unlanded_callsigns,
            "window": self._window_callsigns,
            "all": lambda: None,
        }[str(WindowedConfig.DEVIATION_SCOPE)]()
        conflict_callsigns = (
            self._window_callsigns() if str(WindowedConfig.CONFLICT_SCOPE) == "window" else None
        )
        return dev_callsigns, conflict_callsigns

    @staticmethod
    def _scoped_reward_parts(action_cmds, predicted_world, current_time_s, remaining_wps, scopes) -> RewardParts:
        dev_callsigns, conflict_callsigns = scopes
        return RewardParts(
            deviation=float(
                _deviation_reward_rollout(
                    predicted_world,
                    current_time_s,
                    callsigns=dev_callsigns,
                    horizon_s=float(WindowedConfig.DEVIATION_RAMP_S),
                    baseline=float(WindowedConfig.DEVIATION_FAR_WEIGHT),
                )
            ),
            conflict=float(
                infringement_reward_from_world(predicted_world, current_time_s, callsigns=conflict_callsigns)
            ),
            action=-float(_action_penalty(action_cmds, remaining_wps)),
            goal=0.0,  # the goal bonus is an episode-set construct, like the tier ladder
        )

    def _advance_and_compute_reward(
        self, action_cmds, actor_id: str, action_num: int, remaining_wps: Optional[int] = None
    ):
        assert self.simulator.world is not None and self.current_predicted_world is not None
        # Scopes are taken at DECISION time (before advancing): a flight that lands during this
        # step is still charged for it, and drops out from the next step on.
        scopes = self._reward_scopes()

        # Reselection: the clearances queued at this step go in with this one. The action
        # penalty below still reads this pick's commands only (queued ones paid when queued).
        self.simulator.apply_action_and_advance(
            self._with_queued(action_cmds), time_between_actions=Config.TIME_BETWEEN_ACTIONS
        )
        self._steps += 1
        self.global_steps += 1

        current_estimate = self._predict(self.simulator.world)
        if self._pbrs():
            # Objective only; the shaping is added in step() once Phi(s') is known. A clearance is
            # charged only when it is real: DO_NOTHING returns one command with command=None, which
            # _action_penalty used to count like any other.
            real = int(action_num) != int(Action.DO_NOTHING)
            r_parts = RewardParts(
                deviation=0.0, conflict=0.0,
                action=-self._clearance_cost if real else 0.0,  # a count, not weighted
                goal=0.0,
            )
        else:
            r_parts = self._scoped_reward_parts(
                action_cmds, current_estimate, float(self.simulator.world.age_in_seconds), remaining_wps, scopes
            )

        self._update_render_state_after_advance(r_parts, actor_id, action_num)
        self.current_predicted_world = current_estimate
        return float(r_parts.total()), r_parts

    def _eval_clearance(self, clr: int, callsign: str, remaining_wps: int):
        """Read-only look-ahead used by the eval shield, with the windowed (scoped) reward."""
        from actions.actions import interpret_action_readonly

        cmds = interpret_action_readonly(clr, callsign, self.simulator, self._route_trombone_store)
        world_after = self.simulator.simulate_candidate(self._with_queued(cmds))
        predicted = self._predict(world_after)
        now_s = float(world_after.age_in_seconds)
        r_parts = self._scoped_reward_parts(cmds, predicted, now_s, remaining_wps, self._reward_scopes())
        return float(r_parts.total()), bool(has_near_conflict(predicted, now_s))

    def _violation_penalty(self) -> float:
        """Constant (positive magnitude). Legacy: BASE + a window's worth of forfeited landing
        bonuses. outcome_pbrs: WindowedConfig.PBRS_VIOLATION_PENALTY (or the env's
        "pbrs_violation_penalty"), which exceeds landing the whole window at the worst bracket
        instead of the best, so safety dominates."""
        if self._pbrs():
            return self._pbrs_violation_penalty
        return float(WindowedConfig.VIOLATION_PENALTY_BASE) + float(WindowedConfig.VIOLATION_FORFEIT)

    def _sequence_metrics(self) -> Dict[str, float]:
        """How well the fixed AMAN sequence was kept, over the flights that actually landed.

          aman_order_rate   share of landed flights whose landing rank equals their AMAN rank
                            (both counted among landed flights)
          sequence_swaps    inverted pairs between landing order and AMAN order
          clean_stretch     share of consecutive 10-flight AMAN stretches in which every flight
                            landed and >= 9 were on time -- comparable across stream lengths,
                            unlike "all N on time", which compounds the per-flight rate N times
        """
        on_time_s = float(WindowedConfig.ON_TIME_DEV_S)
        landed = [c for c in sorted(self._eta, key=self._eta.get) if c in self._landing_devs]
        if not landed:
            return {"window_aman_order_rate": 0.0, "window_sequence_swaps": 0.0,
                    "window_clean_stretch_rate": 0.0}
        land_t = {c: self._eta[c] + float(self._landing_devs[c]) for c in landed}
        by_landing = sorted(landed, key=land_t.get)
        in_place = sum(1 for i, c in enumerate(landed) if by_landing[i] == c)
        rank = {c: i for i, c in enumerate(by_landing)}
        seq = [rank[c] for c in landed]
        swaps = sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])
        aman = sorted(self._eta, key=self._eta.get)
        stretches = [aman[k:k + 10] for k in range(0, len(aman) - 9, 10)]
        clean = sum(
            1 for st in stretches
            if all(c in self._landing_devs for c in st)
            and sum(abs(self._landing_devs[c]) <= on_time_s for c in st) >= 9
        )
        return {
            "window_aman_order_rate": float(in_place / len(landed)),
            "window_sequence_swaps": float(swaps),
            "window_clean_stretch_rate": float(clean / len(stretches)) if stretches else 0.0,
        }

    def _build_episode_metrics(self) -> Dict[str, Any]:
        metrics = super()._build_episode_metrics()
        assert self.simulator.world is not None
        n_flights = len(self.simulator.world.aircraft)
        on_time_s = float(WindowedConfig.ON_TIME_DEV_S)
        abs_devs = [abs(d) for d in self._landing_devs.values()]
        n_landed = len(abs_devs)
        n_on_time = sum(1 for d in abs_devs if d <= on_time_s)
        occ = self._window_occupancy or [0]
        metrics.update(
            {
                "window_n_flights": float(n_flights),
                "window_n_landed": float(n_landed),
                # Per-flight success: on-time landings over ALL flights of the scenario, so a bust
                # or a timeout counts every unlanded flight as a miss. This is the scale-free
                # number to compare across scenario sizes; the 10-aircraft all-on-time rate is
                # roughly this to the 10th power.
                "window_on_time_rate": float(n_on_time / n_flights) if n_flights else 0.0,
                "window_on_time_rate_landed": float(n_on_time / n_landed) if n_landed else 0.0,
                "window_mean_abs_landing_dev": float(np.mean(abs_devs)) if abs_devs else 0.0,
                "window_max_abs_landing_dev": float(max(abs_devs)) if abs_devs else 0.0,
                "window_occupancy_mean": float(np.mean(occ)),
                "window_occupancy_max": float(max(occ)),
                # Fraction of steps on which a flight under control sat OUTSIDE the window (more
                # than WINDOW_SIZE ahead of it in the queue): responsible for, but invisible.
                "window_overflow_frac": float(self._window_overflow_steps / max(1, len(occ))),
                "window_landing_bonus_total": float(self._landing_bonus_total),
                # Real clearances issued (DO_NOTHING excluded): the third criterion of the
                # objective, after safety and deviation bracket.
                "window_clearances": float(sum((self._action_counts or {}).values())),
                # Sum of per-flight BRACKET values (landing_value) over landed flights, whatever
                # the reward mode: the second criterion of the objective, on one scale for every
                # run. Unlanded flights (after a bust or a timeout) contribute nothing.
                "window_bracket_score": float(sum(landing_value(abs(d)) for d in self._landing_devs.values())),
                **self._sequence_metrics(),
                # Cooling gap(s) between stitched segments; 0 when not stitched.
                "window_stitch_gap_mean_s": float(np.mean(self.simulator.stitch_gaps_last))
                if self.simulator.stitch_gaps_last else 0.0,
            }
        )
        if self._train_max_early_s is not None:
            metrics["window_train_redraws"] = float(self._train_redraws)  # cumulative, this env
        if self._release_locked:
            # Flights handed off before touchdown, and how far the realised brackets moved from the
            # ones paid at release (0 when the do-nothing prediction of a released flight holds).
            metrics["window_released"] = float(len(self._released))
            metrics["window_release_correction"] = float(self._release_correction)
        if self._multi_pick:
            # Reselection use: zero-time picks taken, and clock steps with two or more clearances.
            metrics["window_again_picks"] = float(self._again_picks)
            metrics["window_multi_pick_steps"] = float(self._multi_pick_steps)
        return metrics
