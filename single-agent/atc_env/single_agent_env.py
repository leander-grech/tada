"""Single-agent ATC environment with autoregressive (aircraft_id, clearance) actions."""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from actions.actions import Action, interpret_action
from actions.rewards import (
    calculate_reward,
    compute_weighted_infringement_total,
    get_aircraft_time_deviations,
    get_time_deviations,
    has_near_conflict,
    tier_potential,
)
from config.config import Config
from config.simulation_config_path import get_config_path
from flight_simulator import Infringement
from gymnasium import spaces
from matplotlib.colors import to_rgba
from models.observations import build_global_observation
from simulator.simulator import Simulator
from utils.infringement_utils import get_world_infringements, is_realised_los, pair_severity
from utils.render_utils import (
    ensure_agent_colors,
    render_frame,
)
from utils.reward_utils import RewardParts
from simulator.route_section_store  import RouteSectionStore


class ATCSingleAgentEnv(gym.Env):
    """Single-agent env exposing Tuple(Discrete(N_aircraft), Discrete(N_clearances))."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config: Dict[str, Any] | None = None, verbose: bool = False):
        super().__init__()
        config = config or {}
        self._verbose = verbose

        self.action = []
        self.obs_matrix = []
        self.extra_steps = 0
        self.episode_no = 0
        self.evaluation_mode = config.get("evaluation_mode", False)
        self._eval_randomize_count = config.get("evaluation_randomize_scenario_count", 0)
        self._eval_randomize_used = 0

        self.difficulty = (
            Config.DEFAULT_EVALUATION_DIFFICULTY
            if self.evaluation_mode
            else Config.DEFAULT_TRAINING_DIFFICULTY
        )

        self._eval_seeds = list(Config.SCENARIO_EVAL_SEEDS)
        self._eval_episode_index = -1

        self.scenario_seed = None
        if self.evaluation_mode:
            self.scenario_seed = int(self._eval_seeds[0])
        elif Config.SCENARIO_TRAIN_SEED is not None:
            vector_env_index = config.get("vector_index", 0)
            self.scenario_seed = Config.SCENARIO_TRAIN_SEED + vector_env_index

        # Explicit override (tests/eval) pins the horizon; otherwise it is sized per scenario
        # in reset() from the latest arrival ETA (see _compute_episode_horizon).
        self._max_steps_override = config.get("max_steps_per_episode", None)
        self.max_steps_per_episode = (
            int(self._max_steps_override)
            if self._max_steps_override is not None
            else int(Config.EPISODE_STEPS)
        )
        self._steps = 0
        self.global_steps = 0

        self._route_trombone_store = RouteSectionStore()

        simulation_config_path = config.get("simulation_config_path", None)
        if not simulation_config_path:
            simulation_config_path = get_config_path(difficulty=self.difficulty)

        if self._verbose: print(f"simulation_config_path = {simulation_config_path.simulation_config_root}")
        self.simulator = Simulator(self.difficulty, simulation_config_path, self._route_trombone_store, self._verbose)

        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(-1.0, 1.0, shape=(int(Config.GLOBAL_DIM),), dtype=np.float32),
                "aircraft": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(int(Config.N_AIRCRAFT_MAX), int(Config.AIRCRAFT_SCALAR_LEN)),
                    dtype=np.float32,
                ),
                "mask_aircraft": spaces.MultiBinary(int(Config.N_AIRCRAFT_MAX)),
                "mask_action_per_ac": spaces.MultiBinary(
                    (int(Config.N_AIRCRAFT_MAX), int(Config.N_CLEARANCES))
                ),
                # Per-aircraft flight plan relative waypoint features
                "flight_plans_rel": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(
                        int(Config.N_AIRCRAFT_MAX),
                        int(Config.MAX_WAYPOINTS),
                        int(Config.FLIGHT_PLAN_WAYPOINT_DIM),
                    ),
                    dtype=np.float32,
                ),
                # Corresponding mask for valid waypoints per aircraft
                "flight_plan_mask": spaces.MultiBinary(
                    (int(Config.N_AIRCRAFT_MAX), int(Config.MAX_WAYPOINTS))
                ),
                "action_histories": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        int(Config.N_AIRCRAFT_MAX),
                        int(Config.MAX_HISTORY),
                        1 + int(Config.N_CLEARANCES),
                    ),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.MultiDiscrete(
            [int(Config.N_AIRCRAFT_MAX), int(Config.N_CLEARANCES)]
        )

        # Reset method attributes
        # self.prev_world = None
        self.initial_predicted_world = None # Frozen at env reset()
        self.current_predicted_world = None # Updated every env step()
        # Reset reward tracking
        self.cumulative_reward = 0.0

        self.last_reward = None # RewardParts(0.0, 0.0, 0.0)
        self._action_history = None         # []
        self.extra_steps = None         # 0
        self._past_tracks = None        # {}
        self._initial_routes = None         # {}
        self._last_actor_id = None      # "-------"
        self._last_action_num = None        # -1
        self._action_counts = None      # per-callsign issued-clearance counts (set in reset)
        self._approach_xy = None        # airport/merge point the routes converge on (set in reset)

        # Render-only: number of samples along the NOOP rollout drawn as each aircraft's
        # predicted trajectory. Costs roughly one extra rollout per RENDERED frame, so it is
        # off unless a caller that actually renders asks for it (see render_policy.py).
        self._render_predict_points = int(config.get("render_predict_points", 0))
        # Half-width (NM) of the 3D approach view around the airport.
        self._render_approach_zoom_nm = float(config.get("render_approach_zoom_nm", 25.0))
        # Top-right render panel: "time" (altitude-vs-time side view of the predicted traffic,
        # with infringements marked) or "3d" (the 3D approach view).
        self._render_approach_view = str(config.get("render_approach_view", "time"))

        self._agent_colors = None
        self.initial_deviation = None
        self.episode_time_horizon = None
        self.initial_infr_total = None
        self.last_end_reason = None
        self._prev_aircraft_order = None

        self.acft_perm = None
        self.episodic_translation_matrix = None
        self._prev_potential = 0.0  # Φ(s_{t-1}) for PBRS; initialised in reset()
        self._last_abs_devs: tuple[float, ...] = ()  # set by _evaluate_success()


    @property
    def prev_world(self):
        return self.simulator.world

    def _select_scenario_seed(self, pinned: Optional[int] = None):
        """Select the appropriate scenario seed based on evaluation mode and episode index.

        ``pinned`` (from ``reset(options={"scenario_seed": N})``) overrides the walk through
        ``SCENARIO_EVAL_SEEDS`` entirely and generates scenario N. This is what makes a single
        scenario addressable -- for replaying one seed, or retrying it several times -- rather
        than only reachable by counting resets from env construction.
        """
        if pinned is not None:
            self.scenario_seed = int(pinned)
            return

        if self.evaluation_mode:
            # Advance episode index for this evaluation run
            self._eval_episode_index = self._eval_episode_index + 1

            fixed_count = len(self._eval_seeds)
            randomize_quota = max(0, int(self._eval_randomize_count))

            if self._eval_episode_index < fixed_count:
                # Use fixed seeds in order first
                self.scenario_seed = int(self._eval_seeds[self._eval_episode_index])
            else:
                # After fixed seeds: randomize up to the quota, then continue round-robin
                if self._eval_randomize_used < randomize_quota:
                    eval_seed_set = set(self._eval_seeds)
                    self.scenario_seed = next(
                        seed
                        for seed in iter(lambda: random.randint(0, 2**31 - 1), None)
                        if seed not in eval_seed_set
                    )
                    self._eval_randomize_used += 1
                else:
                    idx = self._eval_episode_index % fixed_count if fixed_count > 0 else 0
                    self.scenario_seed = int(self._eval_seeds[idx])

    def sample_translation_matrix(self):
        dx = np.random.uniform(-Config.GLOBAL_POS_XY_MAX * 0.5, Config.GLOBAL_POS_XY_MAX * 0.5)
        dy = np.random.uniform(-Config.GLOBAL_POS_XY_MAX * 0.5, Config.GLOBAL_POS_XY_MAX * 0.5)

        T = np.array([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        return T

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)

        options = options or {}
        self._select_scenario_seed(options.get("scenario_seed"))

        self.episode_no += 1
        self._steps = 0

        # An explicit sim_kw wins; otherwise regenerate the scenario for the selected seed.
        # (`.get` on a dict without the key returns None, so `or` is what makes an options
        # dict carrying only `scenario_seed` fall through to the default kwargs.)
        sim_kw = options.get("sim_kw") or dict(
            difficulty=self.difficulty,
            create_world=True,
            seed=self.scenario_seed,
            reset_seed=True,
            eval_mode=self.evaluation_mode,
        )

        self.simulator.reset(
            **sim_kw
        )

        # --- initialize render state ---
        assert self.simulator.world is not None, f"Simulator in reset method is None"

        # Size the episode horizon to this scenario (sets max_steps_per_episode +
        # episode_time_horizon) BEFORE the rollout below, so the NOOP deviation rollout is
        # long enough to reach every aircraft's landing.
        self._compute_episode_horizon()

        # self.prev_world = self.simulator.advance_multiple_steps(
        #     self.max_steps_per_episode, self.simulator.world
        # )
        self.initial_predicted_world = self.current_predicted_world = self.simulator.advance_multiple_steps(
            self.max_steps_per_episode, self.simulator.world
        )
        # Reset reward tracking
        self.cumulative_reward = 0.0

        self.last_reward = RewardParts(0.0, 0.0, 0.0)
        self._action_history = []
        self.extra_steps = 0
        self._past_tracks = {}
        self._initial_routes = {}
        self._last_actor_id = "-------"
        self._last_action_num = -1

        # Stable colors per callsign
        cmap = plt.get_cmap("tab10")
        world_callsigns = [ac.call_sign for ac in self.simulator.world.aircraft]
        self._agent_colors = {
            callsign: to_rgba(cmap(i % 10)) for i, callsign in enumerate(world_callsigns)
        }

        # Per-aircraft count of (non DO_NOTHING) clearances issued so far this episode; shown
        # in the render info panel and reset with every episode.
        self._action_counts = {callsign: 0 for callsign in world_callsigns}

        # Seed past tracks with current positions and initial route names
        for ac in self.simulator.world.aircraft:
            pos = ac.state.position_nm_nm_ft
            self._past_tracks[ac.call_sign] = [(float(pos.x), float(pos.y))]
            route_names: List[str] = [rp.waypoint_name for rp in ac.route] if ac.route else []
            self._initial_routes[ac.call_sign] = route_names

        self._approach_xy = self._resolve_approach_xy()

        # self.initial_deviation = get_time_deviations(self.prev_world)[1]
        self.initial_deviation = get_time_deviations(self.current_predicted_world)[1]
        # episode_time_horizon was set by _compute_episode_horizon() above.
        self.initial_infr_total = compute_weighted_infringement_total(
            # get_world_infringements(self.prev_world),
            get_world_infringements(self.current_predicted_world),
            self.episode_time_horizon / 2.0,
            float(self.simulator.world.age_in_seconds))
        self.last_end_reason = None
        self._near_conflict_violated = False
        # Large => "cleared"; reset to 0 on any step where a near-conflict is present.
        self._steps_since_near_conflict = 10 ** 9

        # seed the PBRS potential from the initial predicted world so the first
        # step's shaping (γΦ(s_1) − Φ(s_0)) correctly measures progress from the start.
        _, _, _phi_init = self._evaluate_success(self.initial_predicted_world)
        self._prev_potential = tier_potential(_phi_init, self._last_abs_devs)

        self.obs_matrix = self.sample_translation_matrix()

        # Random permutation across the full slot range to avoid positional bias
        # self.acft_perm = np.random.permutation(int(Config.N_AIRCRAFT_MAX)).tolist()
        self.acft_perm = list(range(int(Config.N_AIRCRAFT_MAX)))
        obs, self._prev_aircraft_order = build_global_observation(
            self.simulator,
            # self.prev_world,
            self.current_predicted_world,
            self._action_history,
            perm=self.acft_perm,
            route_trombone_store=self._route_trombone_store,
            T_obs = self.obs_matrix
        )

        self._obs = obs

        assert self._prev_aircraft_order is not None and len(self._prev_aircraft_order) > 0

        # print({k: np.shape(v) for k, v in obs.items()})
        # print("OBS TYPE:", type(obs))
        # print("OBS:", obs)
        # print("OBS SHAPE:", getattr(obs, "shape", None))
        # print("OBS DTYPE:", getattr(obs, "dtype", None))
        # print("SPACE:", self.observation_space)

        # assert self.observation_space.contains(obs)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Tuple[int, int] | np.ndarray):  # type: ignore[override]
        assert self.simulator.world is not None, "World is not initialized"

        n_clearances = int(Config.N_CLEARANCES)

        action_arr = np.asarray(action).ravel()
        a_air = int(action_arr[0])
        a_clr = int(action_arr[1])

        # Keep a flat action index for logging.
        self.action = a_air * n_clearances + a_clr

        callsign = self._prev_aircraft_order[a_air]

        # Remaining waypoints of the commanded aircraft *before* advancing: a proxy for
        # degrees of freedom. Fewer remaining waypoints => acting later => higher action cost.
        commanded_ac = self.simulator.world.get_aircraft_by_callsign(callsign)
        remaining_wps = len(commanded_ac.route) if (commanded_ac and commanded_ac.route) else 0

        action_cmds = interpret_action(
            a_clr,
            callsign,
            self.simulator,
            self._route_trombone_store,
        )

        # Advance simulation and compute reward
        reward, r_parts = self._advance_and_compute_reward(
            action_cmds,
            callsign,
            a_clr,
            remaining_wps=remaining_wps,
        )

        # Track whether a near-horizon predicted conflict was ever introduced
        # during the episode (a success precondition is that it never was).
        if has_near_conflict(
            self.current_predicted_world, float(self.simulator.world.age_in_seconds)
        ):
            self._near_conflict_violated = True
            self._steps_since_near_conflict = 0
        else:
            self._steps_since_near_conflict += 1

        terminated = self._check_terminated()
        truncated = self._check_truncated(terminated)

        # --- Per-step PBRS tier potential shaping ---
        # Evaluate tier fractions from the committed NOOP rollout (same world used by the
        # deviation reward and _goal_bonus, so gradients are consistent).  At the terminal
        # absorbing state Φ = 0 by convention, so the shaping at the last real step is
        # γ·0 − Φ(s_T) = −Φ(s_T) and the total shaping per episode telescopes to −Φ(s_0).
        _, _pbrs_tier, _pbrs_metrics = self._evaluate_success(self.current_predicted_world)
        _phi_next = 0.0 if (terminated or truncated) else tier_potential(_pbrs_metrics, self._last_abs_devs)
        self.last_reward.bonus += float(Config.PBRS_GAMMA) * _phi_next - self._prev_potential
        self._prev_potential = _phi_next

        # --- Terminal signals ---
        # Violation penalty is unchanged; replaces the large tier-ladder cliff with a small
        # T5-only confirmation so the critic no longer needs to predict a high-variance lump.
        self._terminal_tier = 0
        if self.last_end_reason in ("separation", "airspace_exit"):
            self.last_reward.bonus -= self._violation_penalty()
        elif self.last_end_reason in ("success", "delayed", "timeout"):
            self._terminal_tier = int(_pbrs_tier)
            if self._terminal_tier == 5:
                self.last_reward.bonus += float(Config.REWARD_SUCCESS_BONUS_TIER5_CONFIRM)

        # Important:
        # self.last_reward may have been modified above, so recompute reward.
        r_parts = self.last_reward
        reward = float(r_parts.total())

        # Build terminal metrics before SB3 can auto-reset the environment.
        episode_metrics = None

        if terminated or truncated:
            episode_metrics = self._build_episode_metrics()

        # Build next observation
        obs, self._prev_aircraft_order = build_global_observation(
            simulator=self.simulator,
            curr_world=self.current_predicted_world,
            global_action_history=self._action_history,
            perm=self.acft_perm,
            route_trombone_store=self._route_trombone_store,
            T_obs=self.obs_matrix,
        )

        self._obs = obs

        if self._prev_aircraft_order is None:
            any_left = any(
                not ac.mission.completed
                for ac in self.simulator.world.aircraft
            )
            print(any_left)

        assert (
                self._prev_aircraft_order is not None
                and len(self._prev_aircraft_order) > 0
        )

        # This info is what the training callback reads.
        info = {
            "action": int(self.action),  # flat aircraft*N_CLEARANCES + clearance (both modes)
            "reward": float(reward),
            "reward_parts": r_parts,

            "callsign": callsign,
            "aircraft_index": int(a_air),
            "clearance": int(a_clr),
            "end_reason": self.last_end_reason,
        }

        if episode_metrics is not None:
            info["episode_metrics"] = episode_metrics
            info["is_success"] = bool(episode_metrics["success"])

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array"):  # type: ignore[override]
        # Match multi-agent overlays using extracted utils
        assert mode == "rgb_array", "Only rgb_array mode is supported for recording"
        assert self.simulator.world is not None
        # assert self.prev_world is not None
        assert self.current_predicted_world is not None
        possible_agents = list({ac.call_sign for ac in self.simulator.world.aircraft})
        self._agent_colors = ensure_agent_colors(possible_agents, self._agent_colors)

        # Currently-achieved success tier (0..5) + fractions, for the health-bar strip.
        _, tier_now, tier_metrics = self._evaluate_success(self.current_predicted_world)

        rgb = render_frame(
            simulator=self.simulator,
            # prev_world=self.prev_world,
            prev_world=self.current_predicted_world,
            possible_agents=possible_agents,
            agent_colors=self._agent_colors,
            past_tracks=self._past_tracks,
            initial_routes=self._initial_routes,
            action_history=self._action_history,
            cumulative_reward=self.cumulative_reward,
            last_reward=self.last_reward,
            last_actor_id=self._last_actor_id,
            last_action_num=self._last_action_num,
            title=self._render_title(),
            tier=int(tier_now),
            tier_metrics=tier_metrics,
            action_counts=self._action_counts,
            approach_xy=self._approach_xy,
            approach_zoom_nm=self._render_approach_zoom_nm,
            predicted_tracks=self.predicted_tracks(self._render_predict_points),
            approach_view=self._render_approach_view,
            side_view=self.side_view_data() if self._render_approach_view == "time" else None,
        )
        return rgb

    # ---------------------- helpers ----------------------

    def _prediction_horizon_steps(self) -> int:
        """Steps the NOOP prediction rollout covers (render only; the windowed env's is receding)."""
        return int(self.max_steps_per_episode)

    def _side_view_highlight(self) -> Optional[set]:
        """Callsigns the side view draws at full strength; None = every aircraft."""
        return None

    def side_view_data(self) -> Dict[str, Any]:
        """Everything the altitude-vs-time side view needs, times relative to now (render only).

        Profiles are sampled once per TIME_BETWEEN_ACTIONS along a do-nothing rollout of the
        same horizon as ``current_predicted_world``, so the events drawn on them are the ones
        the reward and observation see. A flight's touchdown is placed at its predicted landing
        time (the MISS_TIMED_LANDING event, or its AMAN target when within the simulator's
        10 s tolerance), not at the next sample, which could be up to 45 s late.
        """
        from utils.infringement_utils import pair_severity

        assert self.simulator.world is not None
        world = self.simulator.world
        now = float(world.age_in_seconds)
        targets = {str(t.ac_callsign): float(t.eta_s) - now for t in world.arrival_targets}
        pred = self.current_predicted_world
        landing_t: Dict[str, float] = {}
        events = []
        for infr in (pred.infringements if pred is not None else []):
            ts = float(getattr(infr, "timestamp_s", now)) - now
            if isinstance(infr, Infringement.MISS_TIMED_LANDING):
                landing_t[str(infr.aircraft)] = ts
            elif isinstance(infr, Infringement.LOSS_OF_SEPARATION) and ts >= 0.0:
                pair = [str(x) for x in getattr(infr, "aircraft_pair", [])]
                if len(pair) == 2:
                    sev = pair_severity(float(infr.horizontal_separation_nm),
                                        float(infr.vertical_separation_ft))
                    events.append((ts, pair[0], pair[1], float(sev)))

        profiles: Dict[str, List[Tuple[float, float]]] = {}
        landed_at: Dict[str, float] = {}
        for ac in world.aircraft:
            if ac.under_control and not ac.mission.completed:
                profiles[ac.call_sign] = [(0.0, float(ac.state.position_nm_nm_ft.z))]
        w = world
        for _ in range(self._prediction_horizon_steps()):
            w = self.simulator.advance_multiple_steps(1, w)
            t = float(w.age_in_seconds) - now
            for ac in w.aircraft:
                cs = ac.call_sign
                if cs in landed_at:
                    continue
                if ac.mission.completed:
                    if cs in profiles:  # touched down during this interval
                        t_land = landing_t.get(cs, targets.get(cs, t))
                        t_land = min(max(t_land, profiles[cs][-1][0]), t)
                        profiles[cs].append((t_land, 0.0))
                        landed_at[cs] = t_land
                    continue
                if ac.under_control:
                    profiles.setdefault(cs, []).append((t, float(ac.state.position_nm_nm_ft.z)))
        return {
            "profiles": profiles,
            "landed_at": landed_at,
            "targets": targets,
            "events": events,
            "highlight": self._side_view_highlight(),
            "horizon_s": float(self._prediction_horizon_steps() * Config.TIME_BETWEEN_ACTIONS),
        }

    def _render_title(self) -> str:
        """Frame title; subclasses add their own state (the windowed env shows its window)."""
        return f"TADA evaluation (seed = {self.scenario_seed})"

    def _resolve_approach_xy(self) -> Tuple[float, float]:
        """(x, y) of the airport the arrivals converge on, for the 3D approach view.

        ``world.runways`` is empty in the generated scenarios, so the airport is recovered from
        the routes instead: every arrival route ends at the field (MXP on the trombone use
        case), so the centroid of the final waypoints IS the airport, and stays correct if a
        scenario ever mixes runways. Falls back to the origin only when no route has waypoints.
        """
        assert self.simulator.world is not None
        final_pts = []
        for names in self._initial_routes.values():
            if not names:
                continue
            wp = self.simulator.world.waypoints.get(names[-1])
            if wp is not None:
                final_pts.append((float(wp.position_nm.x), float(wp.position_nm.y)))
        if not final_pts:
            return (0.0, 0.0)
        return (
            float(np.mean([p[0] for p in final_pts])),
            float(np.mean([p[1] for p in final_pts])),
        )

    def predicted_tracks(
        self, n_points: int = 0
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """Per-callsign predicted trajectory ``[(x_nm, y_nm, alt_ft), ...]`` under DO_NOTHING.

        This is the same forecast the deviation reward is scored against
        (``current_predicted_world`` is its endpoint), sampled at ``n_points`` intermediate
        states instead of only the final one -- so the render can show where every aircraft is
        *going*, not just where it is.

        The rollout is pure (``simulate_n_next_states`` returns a new world) and costs about
        the same as the one committed rollout per step, which is why it is opt-in: it is only
        worth paying on frames that are actually rendered. ``n_points <= 0`` returns {}.
        """
        assert self.simulator.world is not None
        horizon = int(self.max_steps_per_episode)
        if n_points <= 0 or horizon <= 0:
            return {}

        # Roll the FULL horizon forward from now, matching the committed rollout that
        # `current_predicted_world` and the deviation reward are built from -- a forecast that
        # stopped at the remaining step budget would disagree with what the agent is scored on.
        chunk = max(1, math.ceil(horizon / int(n_points)))
        tracks: Dict[str, List[Tuple[float, float, float]]] = {}
        for ac in self.simulator.world.aircraft:
            if not ac.mission.completed and ac.under_control:
                p = ac.state.position_nm_nm_ft
                tracks[ac.call_sign] = [(float(p.x), float(p.y), float(p.z))]

        landed: set = set()
        world = self.simulator.world
        for _ in range(int(n_points)):
            world = self.simulator.advance_multiple_steps(chunk, world)
            for ac in world.aircraft:
                if ac.call_sign not in tracks or ac.call_sign in landed:
                    continue
                if ac.mission.completed:
                    # It lands this chunk. Close the track on the airport at ground level --
                    # where a completed arrival demonstrably ends -- instead of leaving it
                    # hanging at whatever sample happened to fall before touchdown, and stop
                    # tracking it (the simulator keeps a landed aircraft in the world).
                    landed.add(ac.call_sign)
                    if self._approach_xy is not None:
                        tracks[ac.call_sign].append(
                            (float(self._approach_xy[0]), float(self._approach_xy[1]), 0.0)
                        )
                    continue
                p = ac.state.position_nm_nm_ft
                tracks[ac.call_sign].append((float(p.x), float(p.y), float(p.z)))
        return tracks

    def _compute_episode_horizon(self) -> None:
        """Set ``max_steps_per_episode`` and ``episode_time_horizon`` for the current scenario.

        Per-scenario horizon: size the episode to the latest arrival ETA so aircraft that
        physically arrive beyond the fixed 64-step horizon are not cut off (the dominant cause
        of timeouts). An explicit ``max_steps_per_episode`` override (tests/eval) or
        ``USE_PER_SCENARIO_HORIZON=False`` falls back to the fixed ``EPISODE_STEPS``.
        """
        assert self.simulator.world is not None
        tba = float(Config.TIME_BETWEEN_ACTIONS)
        max_eta = max(
            (float(target.eta_s) for target in self.simulator.world.arrival_targets),
            default=Config.EPISODE_STEPS * tba,
        )
        self.episode_time_horizon = max_eta

        if self._max_steps_override is not None:
            self.max_steps_per_episode = int(self._max_steps_override)
        elif Config.USE_PER_SCENARIO_HORIZON:
            steps = math.ceil(max_eta / tba) + int(Config.HORIZON_STEP_MARGIN)
            steps = min(max(steps, int(Config.MIN_EPISODE_STEPS)), int(Config.MAX_EPISODE_STEPS))
            self.max_steps_per_episode = int(steps)
        else:
            self.max_steps_per_episode = int(Config.EPISODE_STEPS)

    def _eval_clearance(self, clr: int, callsign: str, remaining_wps: int):
        """Read-only evaluation of one clearance from the current world: returns
        (total_reward, creates_near_conflict). Pure — no episode state is mutated.

        LENGTHEN_TROMBONE clearances call route_trombone_store.add_section_to_route()
        inside interpret_action(), which mutates the store.  We deep-copy the store for
        those clearances only so peek_action / peek_clearances never touch the live store.
        """
        from actions.actions import interpret_action_readonly

        cmds = interpret_action_readonly(
            clr, callsign, self.simulator, self._route_trombone_store
        )
        world_after = self.simulator.simulate_candidate(cmds)
        predicted = self.simulator.advance_multiple_steps(
            steps=self.max_steps_per_episode, world=world_after
        )
        r_parts = calculate_reward(
            interpreted_actions=cmds,
            previous_estimation=self.initial_predicted_world,
            current_estimation=world_after,
            initial_deviation=self.initial_deviation,
            initial_infr_total=self.initial_infr_total,
            episode_time_horizon=self.episode_time_horizon,
            current_time_s=float(world_after.age_in_seconds),
            evaluation_mode=self.evaluation_mode,
            predicted_world=predicted,
            action_dof_remaining_wps=remaining_wps,
        )
        conflict = bool(has_near_conflict(predicted, float(world_after.age_in_seconds)))
        return float(r_parts.total()), conflict

    def _remaining_wps(self, callsign: str) -> int:
        ac = self.simulator.world.get_aircraft_by_callsign(callsign)
        return len(ac.route) if (ac and ac.route) else 0

    def peek_action(self, a_air: int, a_clr: int) -> Dict[str, Any]:
        """Read-only one-step look-ahead for a candidate (aircraft, clearance) action.

        Evaluates the proposed clearance and the DO_NOTHING baseline on the *current*
        committed world WITHOUT mutating any episode state (``simulate_n_next_states`` is pure
        and ``interpret_action`` has no side effects on the trombone store). Used by the
        eval-time shield in ``render_policy.py`` to refuse clearances that reduce reward or
        introduce a critical conflict.

        Returns a dict with ``reward_action``, ``reward_noop``, ``reward_delta`` (action minus
        no-op; negative => the clearance makes things worse than doing nothing) and
        ``creates_critical_conflict`` (a near-horizon conflict the action introduces that
        DO_NOTHING would not).
        """
        assert self.simulator.world is not None
        callsign = self._prev_aircraft_order[a_air]
        remaining_wps = self._remaining_wps(callsign)
        r_action, c_action = self._eval_clearance(int(a_clr), callsign, remaining_wps)
        r_noop, c_noop = self._eval_clearance(int(Action.DO_NOTHING), callsign, remaining_wps)
        return {
            "reward_action": r_action,
            "reward_noop": r_noop,
            "reward_delta": r_action - r_noop,
            "creates_critical_conflict": bool(c_action and not c_noop),
            "conflict_action": c_action,
            "conflict_noop": c_noop,
        }

    def peek_clearances(self, a_air: int, clrs) -> Dict[int, Dict[str, Any]]:
        """Batch read-only look-ahead: evaluate several candidate clearances for one aircraft,
        computing the DO_NOTHING baseline ONCE (the next_best shield's hot path). Candidates
        that fail to interpret/simulate are skipped. Returns {clr: {reward_delta,
        creates_critical_conflict}}.
        """
        assert self.simulator.world is not None
        callsign = self._prev_aircraft_order[a_air]
        remaining_wps = self._remaining_wps(callsign)
        r_noop, c_noop = self._eval_clearance(int(Action.DO_NOTHING), callsign, remaining_wps)
        out: Dict[int, Dict[str, Any]] = {}
        for clr in clrs:
            try:
                r, c = self._eval_clearance(int(clr), callsign, remaining_wps)
            except Exception:
                continue
            out[int(clr)] = {
                "reward_delta": r - r_noop,
                "creates_critical_conflict": bool(c and not c_noop),
            }
        return out

    def _advance_and_compute_reward(
        self, action_cmds, actor_id: str, action_num: int, remaining_wps: Optional[int] = None
    ) -> float:
        assert self.simulator.world is not None, "World is not initialized"
        assert self.current_predicted_world is not None, "Prev world is not initialized"

        self.simulator.apply_action_and_advance(
            action_cmds, time_between_actions=Config.TIME_BETWEEN_ACTIONS
        )
        self._steps += 1
        self.global_steps += 1

        # Roll out and compute reward
        current_estimate = self.simulator.advance_multiple_steps(
            steps=self.max_steps_per_episode,
            world=self.simulator.world
        )
        # print(self.initial_deviation)
        r_parts = calculate_reward(
            interpreted_actions=action_cmds,
            previous_estimation=self.initial_predicted_world,
            current_estimation=self.simulator.world,
            initial_deviation=self.initial_deviation,
            initial_infr_total=self.initial_infr_total,
            episode_time_horizon=self.episode_time_horizon,
            current_time_s=float(self.simulator.world.age_in_seconds),
            evaluation_mode=self.evaluation_mode,
            predicted_world=current_estimate,
            action_dof_remaining_wps=remaining_wps,
        )
        reward = float(r_parts.total())

        self._update_render_state_after_advance(r_parts, actor_id, action_num)

        # Commit world and prev_world for next frame
        self.current_predicted_world = current_estimate
        return reward, r_parts

    def _update_render_state_after_advance(
        self, reward_parts: RewardParts, actor_id: str, action_num: int
    ) -> None:
        """Update rendering-related state (tracks, markers, reward overlays).

        This method mirrors the multi-agent renderer's bookkeeping but adapted
        for the single-agent environment data structures.
        """
        assert self.simulator.world is not None

        # Rewards
        self.last_reward = reward_parts
        self.cumulative_reward += float(self.last_reward.total())

        # Tracks
        for ac in self.simulator.world.aircraft:
            if not ac.mission.completed:
                pos = ac.state.position_nm_nm_ft
                self._past_tracks.setdefault(ac.call_sign, []).append((float(pos.x), float(pos.y)))

        # Actor/action (use known selection)
        self._last_actor_id = actor_id
        self._last_action_num = int(action_num)

        # Issued-clearance tally (render only). DO_NOTHING is not a clearance.
        if int(action_num) != int(Action.DO_NOTHING) and self._action_counts is not None:
            self._action_counts[actor_id] = self._action_counts.get(actor_id, 0) + 1

        # Action history marker
        if self._last_actor_id != "-------":
            ac = self.simulator.world.get_aircraft_by_callsign(self._last_actor_id)
            if ac is not None:
                pos_post = ac.state.position_nm_nm_ft
                self._action_history.append(
                    {
                        "call_sign": self._last_actor_id,
                        "action": int(self._last_action_num),
                        "x": float(pos_post.x),
                        "y": float(pos_post.y),
                        "step": int(self._steps),
                        "reward": float(self.last_reward.total()),
                        "reward_parts": (self.last_reward),
                    }
                )

    def _evaluate_success(self, dev_world=None):
        """Evaluate the graduated success tier and the binary success flag.

        Deviations are read from ``dev_world`` (defaults to the live simulator
        world).  Passing the committed NOOP rollout lets partially-complete /
        timeout episodes earn the lower tiers from their *estimated* final
        deviations, while ``all landed`` (tier 5) is always judged on the live
        world.  All tiers also require the relaxed no-near-conflict gate.

        Tiers (highest reached wins, no stacking):
          1: >= 50% of aircraft within ±SUCCESS_TIER1_DEV_S (120 s)
          2: >= 80% within ±SUCCESS_TIER2_DEV_S (100 s)
          3: >= 80% within ±SUCCESS_TIER3_DEV_S (70 s) AND max_dev < TIER3_MAX_DEV_S (200 s)
          4: >= 80% within ±SUCCESS_TIER4_DEV_S (60 s) AND max_dev < TIER4_MAX_DEV_S (120 s)
          5: all landed AND all aircraft within ±SUCCESS_TIER5_DEV_S (60 s)  — "solved"

        T3/T4 add a max_aircraft_dev cap to constrain the tail aircraft.
        Each tier strictly implies all lower tiers.

        Returns (success: bool, tier: int, metrics: dict). ``success`` == (tier == 5).
        """
        assert self.simulator.world is not None
        world = dev_world if dev_world is not None else self.simulator.world

        per_ac = get_aircraft_time_deviations(world)
        abs_devs = [abs(float(v)) for v in per_ac.values()]
        # Stashed for tier_potential(): the continuous PBRS coverage needs the raw
        # per-aircraft deviations, not the bucketed counts. It cannot ride in the returned
        # metrics dict -- that is splatted wholesale into episode_metrics below and every value
        # there has to be a loggable scalar.
        #
        # INVARIANT: this is bound to the world THIS call was given, so tier_potential() must be
        # called immediately after the _evaluate_success() whose metrics it is scoring. Both
        # current call sites do (reset() and step()); do not insert another evaluation between
        # them without re-reading this.
        self._last_abs_devs = tuple(abs_devs)
        n = len(abs_devs)
        total_abs = float(sum(abs_devs))
        max_dev = float(max(abs_devs)) if abs_devs else 0.0

        t1_thresh   = float(Config.SUCCESS_TIER1_DEV_S)
        t2_thresh   = float(Config.SUCCESS_TIER2_DEV_S)
        t3_thresh   = float(Config.SUCCESS_TIER3_DEV_S)
        t4_thresh   = float(Config.SUCCESS_TIER4_DEV_S)
        t5_thresh   = float(Config.SUCCESS_TIER5_DEV_S)
        t1_frac_req = float(Config.SUCCESS_TIER1_FRAC)
        t2_frac_req = float(Config.SUCCESS_TIER2_FRAC)   # also T3, T4
        t3_max_dev  = float(Config.SUCCESS_TIER3_MAX_DEV_S)
        t4_max_dev  = float(Config.SUCCESS_TIER4_MAX_DEV_S)

        n_t1 = sum(1 for d in abs_devs if d <= t1_thresh)
        n_t2 = sum(1 for d in abs_devs if d <= t2_thresh)
        n_t3 = sum(1 for d in abs_devs if d <= t3_thresh)
        n_t4 = sum(1 for d in abs_devs if d <= t4_thresh)
        frac_t1 = float(n_t1 / n) if n else 0.0
        frac_t2 = float(n_t2 / n) if n else 0.0
        frac_t3 = float(n_t3 / n) if n else 0.0
        frac_t4 = float(n_t4 / n) if n else 0.0
        all_under_t5 = bool(n > 0 and all(d <= t5_thresh for d in abs_devs))

        # Conflict gate. Two regimes, selected by Config.SUCCESS_CONFLICT_REALISED_ONLY:
        #
        #   realised-only (current): a conflict only counts if it ACTUALLY happened — some
        #     pair was inside the conflict radius (severity == 1.0, i.e.
        #     within 3 NM horizontally and 650 ft vertically) at a real timestep of the live
        #     world. This is the operational definition of a loss of separation, and it is
        #     the same signal that already drives end_reason="separation".
        #
        #   forecast mode: no near-conflict PREDICTED by the NOOP rollout for the final K steps.
        #     Trips on forecasts, so a trajectory that stayed safe in reality could still be
        #     denied every tier because the rollout was pessimistic.
        clear_steps = int(Config.SUCCESS_NEAR_CONFLICT_CLEAR_STEPS)
        if bool(getattr(Config, "SUCCESS_CONFLICT_REALISED_ONLY", False)):
            had_separation, _ = self._episode_violations()
            no_near = not had_separation
        else:
            no_near = bool(self._steps_since_near_conflict >= clear_steps)
        # 'all landed' is a real-completion fact: always from the live world.
        all_landed = bool(all(ac.mission.completed for ac in self.simulator.world.aircraft))

        tier = 0
        if no_near:
            if frac_t1 >= t1_frac_req:                              # T1
                tier = 1
            if frac_t2 >= t2_frac_req:                              # T2
                tier = 2
            if frac_t3 >= t2_frac_req and max_dev < t3_max_dev:     # T3
                tier = 3
            if frac_t4 >= t2_frac_req and max_dev < t4_max_dev:     # T4
                tier = 4
            if all_landed and all_under_t5:                          # T5
                tier = 5

        success = bool(tier == 5)

        metrics = {
            "all_landed": float(all_landed),
            "n_aircraft": float(n),
            "tier": float(tier),
            "n_under_tier1": float(n_t1),
            "n_under_tier2": float(n_t2),
            "n_under_tier3": float(n_t3),
            "n_under_tier4": float(n_t4),
            "frac_under_tier1": frac_t1,
            "frac_under_tier2": frac_t2,
            "frac_under_tier3": frac_t3,
            "frac_under_tier4": frac_t4,
            "all_under_tier5": float(all_under_t5),
            "total_abs_dev": total_abs,
            "max_aircraft_dev": max_dev,
            "no_near_conflicts": float(no_near),
            # The gate that is NOT in force is still logged, so the two regimes can be
            # compared directly on the same episodes.
            "no_near_conflicts_predicted": float(
                self._steps_since_near_conflict >= clear_steps
            ),
            "ever_near_conflict": float(self._near_conflict_violated),
            "steps_since_near_conflict": float(
                min(self._steps_since_near_conflict, int(self.max_steps_per_episode))
            ),
        }
        return success, tier, metrics

    def _episode_violations(self):
        """Realised hard violations accumulated over the episode (read from the live world).

        ``world.infringements`` accumulates every infringement raised since reset, so at any
        step it captures whatever has actually happened so far. Returns
        (had_separation, had_airspace_exit):

          * had_separation: some pair was inside BOTH 3 NM and 500 ft at a real timestep of
            the live world -- i.e. ``severity == 1.0``, the operational loss of separation.
            Purely predicted conflicts never count here, however imminent: that is the whole
            point of the realised-only gate.
          * had_airspace_exit: any aircraft left the airspace under control.

        (mission.failed is never set by the simulator, so it is not a possible reason.)
        """
        assert self.simulator.world is not None
        world = self.simulator.world

        # Single early-exit pass over the raw list rather than get_world_infringements(), which
        # builds a per-aircraft dict of every record before we look at any of them.
        #
        # This runs EVERY step now (from _check_terminated), and world.infringements grows
        # monotonically through an episode, so the dict-building version was quadratic in
        # episode length: measured 15 us at 3 records, 136 us at 24, and episodes lengthen to
        # 85+ as the policy stops busting out early. Small next to a ~27 ms env step, but it
        # was pure waste and it grew in exactly the wrong direction.
        had_separation = False
        had_airspace_exit = False
        for infr in world.infringements:
            if not had_airspace_exit and isinstance(infr, Infringement.AIRSPACE_EXIT):
                had_airspace_exit = True
            elif not had_separation and isinstance(infr, Infringement.LOSS_OF_SEPARATION):
                if is_realised_los(
                    pair_severity(
                        getattr(infr, "horizontal_separation_nm", 0.0),
                        getattr(infr, "vertical_separation_ft", 0.0),
                    )
                ):
                    had_separation = True
            if had_separation and had_airspace_exit:
                break
        return bool(had_separation), bool(had_airspace_exit)

    def _min_separation_nm(self) -> float:
        """Closest horizontal approach reached so far this episode, in NM.

        Only pairs the simulator reported (inside the 10 NM detection radius) are visible,
        so this saturates at 10.0 when nothing ever got close. Diagnostic only.
        """
        assert self.simulator.world is not None
        closest = float(Config.DETECTION_SAFEZONE_RADIUS_NM)
        for lst in get_world_infringements(self.simulator.world).values():
            for t in lst:
                if t.horizontal_nm < closest:
                    closest = float(t.horizontal_nm)
        return closest

    def _raw_infringement_counts(self):
        """Counts of each raw infringement variant accumulated over the episode."""
        assert self.simulator.world is not None
        counts = {"LOSS_OF_SEPARATION": 0, "AIRSPACE_EXIT": 0, "MISS_TIMED_LANDING": 0}
        for infr in self.simulator.world.infringements:
            if isinstance(infr, Infringement.LOSS_OF_SEPARATION):
                counts["LOSS_OF_SEPARATION"] += 1
            elif isinstance(infr, Infringement.AIRSPACE_EXIT):
                counts["AIRSPACE_EXIT"] += 1
            elif isinstance(infr, Infringement.MISS_TIMED_LANDING):
                counts["MISS_TIMED_LANDING"] += 1
        return counts

    def _refine_end_reason(self, base):
        """Upgrade a terminal reason to reflect realized hard violations.

        A loss of separation or airspace exit takes priority over the landing/timeout
        outcome (you did not 'succeed' if you broke separation). Returns ``base`` unchanged
        when the episode is not ending (base is None) so mid-episode steps stay un-classified.
        Possible end reasons: separation, airspace_exit, timeout, success, delayed.
        """
        if base is None:
            return None
        had_separation, had_airspace_exit = self._episode_violations()
        if had_separation:
            return "separation"
        if had_airspace_exit:
            return "airspace_exit"
        return base

    def _check_terminated(self) -> bool:
        """Terminate on a realised hard violation, or when every aircraft has landed.

        The violation check comes FIRST and ends the episode on the spot. A loss of
        separation is the worst outcome in this domain, so the agent does not get to keep
        playing (and keep banking deviation reward) after causing one -- it is an absorbing
        failure, scored by ``_violation_penalty()``.
        """
        assert self.simulator.world is not None

        had_separation, had_airspace_exit = self._episode_violations()
        if had_separation or had_airspace_exit:
            self.last_end_reason = "separation" if had_separation else "airspace_exit"
            return True

        any_left = any(not ac.mission.completed for ac in self.simulator.world.aircraft)

        reason = None
        if not any_left:
            # All aircraft have landed: success if the top tier is reached,
            # otherwise the episode merely completed ("delayed").
            success, _, _ = self._evaluate_success()
            reason = "success" if success else "delayed"

        self.last_end_reason = self._refine_end_reason(reason)

        return reason is not None

    def _violation_penalty(self) -> float:
        """Terminal penalty for an absorbing failure, as a POSITIVE magnitude.

            penalty = BASE + PER_REMAINING_STEP * remaining_steps

        The second term is what stops early termination from being an escape hatch. Without
        it, an agent stuck in an episode with a negative expected step reward could improve
        its return by deliberately busting separation; with it, ending early forfeits at
        least as much as continuing could ever have earned.

        If ``REWARD_VIOLATION_RETURN_FLOOR`` is positive, strict-domination mode applies
        instead: the penalty is whatever it takes to land the episode's TOTAL return on
        exactly ``-floor``, so every violation scores identically and worst. See the config
        note for why that is not the default.
        """
        floor = float(Config.REWARD_VIOLATION_RETURN_FLOOR)
        if floor > 0.0:
            # cumulative_reward already includes this step's parts and the PBRS shaping.
            return float(self.cumulative_reward) + floor

        remaining = max(0, int(self.max_steps_per_episode) - int(self._steps))
        return float(Config.REWARD_VIOLATION_PENALTY) + float(
            Config.REWARD_VIOLATION_PER_REMAINING_STEP
        ) * remaining

    def _check_truncated(self, terminated) -> bool:
        assert self.simulator.world is not None

        hit_time_limit = self._steps >= self.max_steps_per_episode

        reason = None
        if not terminated:
            if hit_time_limit:
                reason = "timeout"

            self.last_end_reason = self._refine_end_reason(reason)

        return reason is not None


    @staticmethod
    def _get_infringement_reason(infringement) -> str:
        """
        Robustly extract an infringement reason/type/name.
        """
        for attr in [
            "reason",
            "type",
            "infringement_type",
            "name",
            "kind",
            "category",
        ]:
            value = getattr(infringement, attr, None)

            if value is None:
                continue

            if hasattr(value, "name"):
                return str(value.name)

            return str(value)

        return infringement.__class__.__name__

    def _build_episode_metrics(self) -> Dict[str, Any]:
        """
        Build terminal episode metrics from the real simulator world.

        This is called inside step() before SB3 can auto-reset the env.
        """
        assert self.simulator.world is not None

        world = self.simulator.world

        # =====================================================
        # Final delay / deviation
        # =====================================================

        total_dev = get_time_deviations(world)
        final_dev = float(total_dev[1]) if total_dev else 0.0
        final_abs_dev = abs(final_dev)

        # =====================================================
        # Final infringements + reasons
        # =====================================================

        infr_map = get_world_infringements(world)

        threshold = float(Config.LOS_SEVERITY_THRESHOLD) - 1e-9

        # Severity sum / count of *severe* separation losses (normalized >= threshold).
        final_infr = 0.0
        severe_separation_count = 0
        for infringement_list in infr_map.values():
            for infringement in infringement_list:
                value = float(getattr(infringement, "value", 0.0))
                if value >= threshold:
                    final_infr += value
                    severe_separation_count += 1

        # Accurate per-variant counts from the raw world infringements (the get_world_infringements
        # view only surfaces LOSS_OF_SEPARATION, so it cannot label airspace exits / miss-timed).
        infringement_reasons = self._raw_infringement_counts()
        had_separation, had_airspace_exit = self._episode_violations()

        # =====================================================
        # Success (AND of all criteria) + detailed sub-metrics
        # =====================================================

        success, _, success_metrics = self._evaluate_success()

        return {
            "deviation_final": float(final_dev),
            "deviation_abs_final": float(final_abs_dev),
            "infringement_total_final": float(final_infr),
            "infringement_count_final": float(severe_separation_count),
            "infringement_reasons": infringement_reasons,
            "had_separation": bool(had_separation),
            "had_airspace_exit": bool(had_airspace_exit),
            "success": bool(success),
            # Graduated tier that actually drove the terminal bonus (measured on
            # the committed rollout). Falls back to the live-world tier above.
            "bonus_tier": float(getattr(self, "_terminal_tier", success_metrics["tier"])),
            "end_reason": self.last_end_reason,
            "episode_steps": int(self._steps),
            # Closest horizontal approach actually reached (NM), saturating at the 10 NM
            # detection radius. With early termination on a bust, episode length alone
            # cannot distinguish "got safer" from "died sooner" -- this can.
            "min_separation_nm": float(self._min_separation_nm()),
            # Fraction of the allotted horizon that was actually flown. 1.0 for a full
            # episode, small for an early bust.
            "horizon_used_frac": float(
                self._steps / max(1, int(self.max_steps_per_episode))
            ),
            "violation_penalty": float(
                self._violation_penalty()
                if self.last_end_reason in ("separation", "airspace_exit")
                else 0.0
            ),
            "scenario_seed": self.scenario_seed,
            **{f"success_{k}": v for k, v in success_metrics.items()},
        }
