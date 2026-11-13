# pylint: disable=too-many-lines
"""Module containing the ATC environment class"""

import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
from flight_simulator import MissionDifficulty
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from actions.action_mask import detect_airport_from_route
from actions.actions import (
    ACTION_DEFINITIONS,
    ActionRecord,
)
from actions.reward_parts import RewardParts
from actions.rewards import (
    compute_action_damping_penalty,
    compute_speed_damping_penalty,
    reward_count_scale,
    reward_options,
    saved_reward_options,
)
from atc_env.action_execution import ActionExecutionMixin
from atc_env.agent_selection import AgentSelectionMixin, SelectionNoopHoldoff
from atc_env.dones import DonesMixin
from atc_env.env_config import observation_settings_from_config, resolve_env_settings
from atc_env.environment_helpers import (
    capture_aircraft_positions,
    distribute_step_reward,
    get_action_selection_mode,
)
from atc_env.initial_rollout import InitialRolloutMixin
from atc_env.observation_builder import ObservationBuilderMixin
from atc_env.outcome_state import OutcomeState
from atc_env.rendering import EnvRenderingMixin
from atc_env.scenario_source import ScenarioSourceMixin
from atc_env.sector_selection import (
    SelectedGroupIndex,
)
from common.hashing import FrozenDict
from config.simulation_config_path import SimulationConfigPath, get_config_path
from models.observation_settings import ObservationSettings
from models.observation_space import build_observation_space
from outcomes import vocabulary
from simulator.route_section_store import RouteSectionStore
from simulator.simulator import Simulator


def _context_index(config: Any, name: str) -> int:
    """Read one runner/vector index that RLlib may expose as attribute or key.

    ``EnvContext`` subclasses ``dict`` but keeps ``worker_index`` and
    ``vector_index`` as attributes, so a plain ``.get()`` lookup would default
    to 0 for every runner. Check attributes first, then keys for envs built
    straight from a plain config dict.
    """
    value = getattr(config, name, None)
    if value is None:
        value = config.get(name)
    return int(value) if value is not None else 0


# pylint: disable=too-many-instance-attributes
class ATCEnv(
    ScenarioSourceMixin,
    ObservationBuilderMixin,
    DonesMixin,
    ActionExecutionMixin,
    AgentSelectionMixin,
    InitialRolloutMixin,
    EnvRenderingMixin,
    MultiAgentEnv,
):
    """ATC environment class"""

    # Pre-generate all possible agent IDs for RLlib registration.
    # Training agents are bounded by the fixed scenario configuration, while
    # evaluation may use a separate larger cap.
    _IMPORT_SETTINGS = resolve_env_settings().env
    _MAX_POSSIBLE_AIRCRAFT = max(
        int(_IMPORT_SETTINGS.max_aircraft_count_training),
        _IMPORT_SETTINGS.max_aircraft_count_eval,
        _IMPORT_SETTINGS.max_aircraft_count_inference,
    )
    _ALL_POSSIBLE_AGENTS = [f"TEST{i:03d}" for i in range(1, _MAX_POSSIBLE_AIRCRAFT + 1)]
    # Planning forecasts look at least this far ahead, even past the episode end.
    forecast_min_lookahead_s = 0

    # Override parent class type hints to use str instead of Hashable.
    # In our environment, all agent IDs are aircraft callsigns (strings).
    agents: list[str]
    possible_agents: list[str]

    _policy_time_limit_steps = None

    @property
    def policy_time_limit_steps(self) -> int:
        """Decisions before the time limit; evaluation may extend it past the saved
        episode length. Observations, forecasts and auto-solve keep
        ``max_steps_per_episode``, so every step before the saved limit is unchanged."""
        override = self._policy_time_limit_steps
        return self.max_steps_per_episode if override is None else override

    def __init__(self, config=None):
        super().__init__()

        config = config or {}
        self.settings = resolve_env_settings()

        self.evaluation_mode = config.get("evaluation_mode", False)
        self._eval_randomize_count = config.get("evaluation_randomize_scenario_count", 0)
        self._eval_randomize_used = 0
        self._airport = config.get("airport")

        from atc_env.completed_scenario import CompletedScenarioCollector

        self.completed_collector = CompletedScenarioCollector()
        self.last_completed_result = None
        self.attempt_id = None
        self._next_attempt_id = None
        self._next_attempt_seed = None
        self._has_reset_once = False
        # RLlib passes an `EnvContext`, a dict subclass that carries the runner
        # and vector index as *attributes* rather than keys, so a plain
        # `config.get("worker_index")` never sees them and every runner would
        # look like the local one (index 0) — collapsing the per-runner
        # scenario streams onto each other.
        self._worker_index = _context_index(config, "worker_index")
        self._vector_index = _context_index(config, "vector_index")
        self._training_stream: Any = None
        self._training_payload = None
        self._inference_checkpoint = None
        self._runtime_activation_required = config.get("require_runtime_activation", False)
        self._runtime_activated = False
        self._scenario_artifact_path: Path | None = None
        self._scenario_config_root: Path | None = None
        self.consumed_semantic_digest: str | None = None
        self.consumed_config_fields: tuple[str, ...] = ()
        if not self.evaluation_mode and {"training_trial_slot", "seed_generation"} & set(config):
            raise ValueError(
                "PB2 training_trial_slot/seed_generation were removed; supply a reserved resolved_env contract"
            )

        self.difficulty = (
            self.settings.env.default_evaluation_difficulty
            if self.evaluation_mode
            else self.settings.env.default_training_difficulty
        )

        if "difficulty" in config:
            explicit_difficulty = getattr(MissionDifficulty, config["difficulty"], None)
            if explicit_difficulty is None:
                raise ValueError("unsupported explicit scenario difficulty")
            self.difficulty = explicit_difficulty

        # Bound runs replace this unbound fallback with their immutable scenario count.
        # No callback stage or jitter changes a resolved training scenario.
        self.max_aircraft_count = config.get(
            "max_aircraft_count",
            (
                self.settings.env.max_aircraft_count_eval
                if self.evaluation_mode
                else int(self.settings.env.max_aircraft_count_training)
            ),
        )
        self.scenario_start_time_offset_max_s = int(
            config.get(
                "scenario_start_time_offset_max_s",
                self.settings.env.scenario_start_time_offset_max_s,
            )
        )
        self._policy_time_limit_steps = config.get("policy_time_limit_steps")
        # Allow overriding via env_config (e.g., evaluation runs)
        self.max_steps_per_episode = config.get(
            "max_steps_per_episode", self.settings.env.episode_steps
        )

        # Deviation bound for the terminal "metrics satisfactory" check, i.e. the
        # window inside which an arrival counts as solved.  Offline evaluation may
        # grade against a wider window than training did (``evaluate
        # --solve-threshold``).  ``success_time_dev_s`` is not an observation
        # input: observations, reward coefficients and candidate selection keep
        # using ``SATISFACTORY_TIME_DEV``.  The override is consumed by
        # ``_compute_dones`` for terminal status and end reason, and that reason
        # selects the terminal reward branch in ``step()``.  Returns and step
        # counts only compare across episodes graded with the same window.
        self.success_time_dev_s = float(
            config.get(
                "success_time_dev_s",
                self.settings.env.operational_success_s
                if self.evaluation_mode
                else self.settings.env.satisfactory_time_dev,
            )
        )
        if not math.isfinite(self.success_time_dev_s) or self.success_time_dev_s <= 0.0:
            raise ValueError(
                "success_time_dev_s must be a positive finite number of seconds, "
                f"got {self.success_time_dev_s!r}"
            )

        if not self.evaluation_mode or "resolved_env" in config:
            self._configure_training_contract(config)

        self.feature_settings = (
            ObservationSettings.from_features(self._training_payload["features"])
            if self._training_payload is not None
            else observation_settings_from_config()
        )
        self.reward_settings = (
            saved_reward_options(self._training_payload["reward"])
            if self._training_payload is not None
            else reward_options(self.settings.core.reward)
        )
        self.lockout_threshold_s = (
            self._training_payload["thresholds"]["lockout_s"]
            if self._training_payload is not None
            else float(self.settings.env.lockout_deviation_threshold_s)
        )
        # Register all possible agents with RLlib upfront (the bound scenario determines capacity)
        self.possible_agents = [
            f"TEST{i:03d}"
            for i in range(1, max(len(self._ALL_POSSIBLE_AGENTS), self.max_aircraft_count) + 1)
        ]

        self._eval_seeds = (
            []
            if self._training_payload is not None and "evaluation_seeds" not in config
            else [
                int(seed)
                for seed in config.get("evaluation_seeds", self.settings.env.scenario_eval_seeds)
            ]
        )
        # Each env runner walks `_eval_seeds` with its own index, so fanning
        # evaluation across runners would replay the same scenarios on every
        # runner.  Sharding by worker index gives the runners disjoint slices
        # whose union is the full manifest. Training uses reserved phase/runner/vector streams.
        if self._inference_checkpoint and self._eval_seeds:
            self._authorize_evaluation_seeds(self._eval_seeds)
            self.inference_input_binding = FrozenDict(
                {"purpose": "disjoint_operational_manifest", "seeds": self._eval_seeds}
            )
        shard_count = int(config.get("evaluation_seed_shards", 0) or 0)
        if shard_count > 0 and self._worker_index > shard_count:
            raise ValueError(
                f"worker_index={self._worker_index} exceeds "
                f"evaluation_seed_shards={shard_count}; remote worker indices must be "
                "between 1 and the configured shard count"
            )
        if shard_count > 1 and self._worker_index > 0:
            self._eval_seeds = self._eval_seeds[
                (self._worker_index - 1) % shard_count :: shard_count
            ]
        if self.evaluation_mode and self._training_payload is None and not self._eval_seeds:
            raise ValueError("evaluation_seeds must contain at least one seed")
        self._eval_episode_index = -1
        # Evaluation cursor guard: only a scenario that reached a terminal reason
        # may consume the next manifest seed (see `_select_evaluation_seed`).
        self._scenario_finished = False
        worker_vector_index = self._cantor_pair(self._worker_index, self._vector_index)
        self._eval_random = (
            None
            if self._training_payload is not None
            else random.Random(int(self.settings.env.seed) + worker_vector_index)
        )

        self.scenario_seed = None
        if self.evaluation_mode and self._training_payload is None:
            self.scenario_seed = int(self._eval_seeds[0])

        self._route_trombone_store = RouteSectionStore()
        self._selected_group_index: SelectedGroupIndex | None = None
        if get_action_selection_mode(self.settings.core.selection) == "sector_based":
            self._selected_group_index = self._resolve_selected_group_index()

        simulation_config_path_raw = config.get("simulation_config_path")
        if self._training_payload is not None:
            simulation_config_path_raw = SimulationConfigPath(str(self._scenario_config_root))
        simulation_config_path: SimulationConfigPath
        if simulation_config_path_raw is None:
            simulation_config_path = get_config_path()
        else:
            simulation_config_path = cast(SimulationConfigPath, simulation_config_path_raw)

        self.outcome_state = OutcomeState(self.settings.core.safety)
        self.noop_verification_max_seconds = (
            self._training_payload["scenario"]["noop_verification_max_seconds"]
            if self._training_payload is not None
            else int(config.get("noop_verification_max_seconds", 14400))
        )
        self.simulator = Simulator(
            self.difficulty,
            simulation_config_path,
            self._route_trombone_store,
            settings=self.settings.core.simulator,
            outcome_owner=self.outcome_state,
        )
        self.episode_max_aircraft_count = self.max_aircraft_count
        if self.evaluation_mode and self._training_payload is None:
            self.simulator.reset(
                difficulty=self.difficulty,
                create_world=True,
                seed=self.scenario_seed,
                eval_mode=True,
                max_aircraft_count=self.episode_max_aircraft_count,
                start_time_offset_max_s=self.scenario_start_time_offset_max_s,
                episode_steps=self.max_steps_per_episode,
            )
        self.prev_world = None

        # Initialize last_end_reason (cleared in step(), NOT in reset(), so callbacks can read it)
        self.last_end_reason = None

        # Pending world for external injection (used by InferenceEngine)
        self._pending_world = None

        if self.evaluation_mode and self._training_payload is None:
            self.reset()
        elif "training_stream_state" in config:
            self.set_training_stream_state(config["training_stream_state"])

    @property
    def arrival_targets(self):
        return self.outcome_state.targets

    @property
    def outcome_ledger(self):
        return self.outcome_state.ledger

    @property
    def last_outcome(self):
        return self.outcome_state.last_outcome

    def _unresolved_failure(self, reason) -> bool:
        """A policy end whose no-op verification or native mission proves a hazard."""
        outcome = self.last_outcome
        assert outcome is not None, "dones classify the outcome before rewards"
        return (
            reason in (vocabulary.UNRESOLVED, vocabulary.TIMEOUT)
            and outcome.kind == vocabulary.UNRESOLVED
            and bool({"verification_hazard", "native_mission_failed"} & set(outcome.reasons))
        )

    @property
    def last_verification(self):
        return self.outcome_state.last_verification

    @property
    def forecast_coverage(self):
        return self.outcome_state.forecast_snapshot

    def _forecast(self, world, policy_steps):
        return self.outcome_state.forecast(
            world, max(0, policy_steps) * int(self.settings.env.time_between_actions)
        )

    def _planning_steps(self, step_counter: int | None = None) -> int:
        """Policy steps the planning forecast covers: the rest of the episode, or the minimum
        lookahead when longer, so conflicts just past the episode end stay visible."""
        step = self.step_counter if step_counter is None else step_counter
        remaining = self.max_steps_per_episode - step
        minimum = math.ceil(self.forecast_min_lookahead_s / self.settings.env.time_between_actions)
        return max(0, remaining, minimum)

    def _configure_training_contract(self, config: dict) -> None:
        raise ValueError(
            "training requires an explicit reserved-manifest resolved_env contract "
            "(construct training.env.TrainingATCEnv)"
        )

    def _reserved_seeds(self) -> frozenset[int]:
        """Frozen benchmark reservations; a standalone env reserves nothing."""
        return frozenset()

    def _check_contract_before_reset(self) -> None:
        """Re-verify the bound contract before generation; standalone envs have none."""

    def _journal_training_scenario(self, status: str) -> None:
        """Journal one scenario identity event; standalone envs keep no journal."""

    def get_observation_space(  # ty: ignore[invalid-method-override]
        self, agent_id: str
    ):  # pylint: disable=unused-argument
        return build_observation_space(self.feature_settings)

    def get_action_space(  # ty: ignore[invalid-method-override]
        self, agent_id: str
    ):  # pylint: disable=unused-argument
        return gym.spaces.Discrete(len(ACTION_DEFINITIONS))

    def reset(self, *, seed=None, options=None):
        if getattr(self, "_runtime_activation_required", False) and not self._runtime_activated:
            raise RuntimeError("native generation requires verified runtime activation")
        self._check_contract_before_reset()
        # If reset() is called while an episode was in progress (step_counter > 0)
        # but no end_reason was set, the episode was interrupted externally (e.g., by
        # RLlib's evaluation duration limit). Set a fallback reason so callbacks can
        # detect this properly.
        if getattr(self, "last_end_reason", None) is None and getattr(self, "step_counter", 0) > 0:
            self.last_end_reason = "interrupted"

        from uuid import uuid4

        from atc_env.completed_scenario import capture_environment

        if self._has_reset_once and self.step_counter > 0 and not self._scenario_finished:
            capture_environment(self, complete=False, error="reset_interrupted")
        super().reset(seed=seed)

        # Preserve caller-seeded RNG state when RLlib doesn't provide a seed.
        if seed is not None and self._pending_world is not None:
            random.seed(seed)

        if self.evaluation_mode and self._next_attempt_seed is not None:
            if not self._inference_checkpoint:
                raise ValueError("explicit attempt seeds require authenticated inference")
            self._authorize_evaluation_seeds([self._next_attempt_seed])
            self.scenario_seed = int(self._next_attempt_seed)
            self._next_attempt_seed = None
        elif self.evaluation_mode:
            self.scenario_seed = self._select_evaluation_seed()
        else:
            self.scenario_seed = self._next_training_scenario_seed()
        self.attempt_id = self._next_attempt_id or uuid4().hex
        self._next_attempt_id = None
        self._scenario_finished = False

        self.episode_max_aircraft_count = self._resolve_episode_max_aircraft_count()
        self._journal_training_scenario("drawn")
        if self._pending_world is not None:
            # Use externally injected world (from InferenceEngine)
            world = self._pending_world
            self._pending_world = None
            world = self.simulator.prepare_world_for_episode(world)
            self.simulator.reset(
                difficulty=self.difficulty,
                world=world,
                world_prepared=True,
                eval_mode=True,
                max_aircraft_count=self.max_aircraft_count,
            )
        else:
            self.simulator.reset(
                difficulty=self.difficulty,
                create_world=True,
                seed=self.scenario_seed,
                reset_seed=True,
                eval_mode=self.evaluation_mode,
                max_aircraft_count=self.episode_max_aircraft_count,
                start_time_offset_max_s=self.scenario_start_time_offset_max_s,
                episode_steps=self.max_steps_per_episode,
            )
        self._journal_training_scenario("generated")
        self._has_reset_once = True

        assert self.simulator.world is not None, "Simulator world is not initialized"
        self.step_counter = 0
        self.last_acted_agent: str | None = None
        self.consecutive_acts: dict[str, int] = {}
        self.stage_cooldown_by_agent: dict[str, int] = {}
        self.recent_noop_holdoff_by_agent: dict[str, SelectionNoopHoldoff] = {}
        self.last_non_zero_action_time_s: dict[str, float] = {}
        self.last_speed_action_time_s: dict[str, float] = {}
        self._last_action_mask_counts: dict[str, int] = {}
        self._locked_out_agents: set[str] = set()
        # Per-step caches — invalidated at the top of step().
        self._conflict_override_cache: set[str] | None = None
        self._step_deviation_proxies: dict[str, float] | None = None
        self._step_selection_infr_map: dict[str, list] | None = None
        self._step_prev_world_all_infr: dict[str, list] | None = None

        # Reset reward tracking
        self.cumulative_reward = 0.0
        self.episode_reward_components = dict.fromkeys(
            ("deviation", "conflict", "action", "bonus"), 0.0
        )
        self.last_reward = RewardParts(0.0, 0.0, 0.0)
        self.last_step_action_nums: dict[str, int] = {}
        self.last_success_margin_bonus: float = 0.0
        self.failed_target_callsigns: tuple[str, ...] = ()
        self.last_observation_count: int = 0
        self.last_blocked_observation_count: int = 0
        self.last_controlled_candidate_count: int = 0
        self.last_outside_excluded_count: int = 0
        # NOTE: Do NOT clear last_end_reason here - it must persist through reset()
        # so that on_episode_end callbacks can read it. It's cleared in step() instead.
        self._action_history: list[ActionRecord] = []
        self._pending_returns: dict[str, deque[str]] = defaultdict(deque)

        self.prev_world = self._prepare_initial_rollout()

        # Agent IDs are the aircraft callsigns (possible_agents is fixed at init for RLlib)
        assert self.simulator.world is not None, "Simulator world is not initialized"
        self.agents = sorted(ac.call_sign for ac in self.simulator.world.aircraft)
        self.initial_agent_count = len(self.arrival_targets)
        current_time_s = float(self.simulator.world.age_in_seconds)
        damping_offset_s = float(
            self.reward_settings["action_damping_steps"] * self.settings.env.time_between_actions
        )
        self.last_non_zero_action_time_s = {
            a: current_time_s - damping_offset_s for a in self.agents
        }
        speed_damping_offset_s = float(
            self.reward_settings["speed_damping_steps"] * self.settings.env.time_between_actions
        )
        self.last_speed_action_time_s = {
            a: current_time_s - speed_damping_offset_s for a in self.agents
        }
        self.stage_cooldown_by_agent = {}
        self.recent_noop_holdoff_by_agent = {}
        self._last_action_mask_counts = {}
        self._aircraft_airports: dict[str, str | None] = (
            {a: self._airport for a in self.agents}
            if self._airport is not None
            else {
                ac.call_sign: detect_airport_from_route(
                    ac.route, self.settings.core.action.custom_action_mask_by_airport
                )
                for ac in self.simulator.world.aircraft
            }
        )

        # Reset render-time state
        self._past_tracks = {}
        self._initial_routes = {}
        self._sector_polygons: list[list[tuple[float, float]]] = []
        self._sector_polygon_colors: list[tuple[float, float, float, float]] = []

        self._agent_color_order = tuple(self.agents)

        next_agents_to_act = self._get_next_agents_rr()
        self._agents_allowed_to_act = set(next_agents_to_act)
        observe_list = list(next_agents_to_act)
        if not observe_list and self.agents:
            # Keep episode alive in sector mode when selected groups have no current candidates.
            observe_list = [self.agents[0]]
        self._record_observation_selection_metrics(observe_list, self._agents_allowed_to_act)
        observations = self._build_observations(observe_list)
        self._record_last_action_mask_counts(observations)

        # Initialize past tracks with current positions
        for ac in self.simulator.world.aircraft:
            pos = ac.state.position_nm_nm_ft
            self._past_tracks[ac.call_sign] = [(float(pos.x), float(pos.y))]
            self._initial_routes[ac.call_sign] = (
                [rp.waypoint_name for rp in ac.route] if ac.route else []
            )
        self._sector_polygons, self._sector_polygon_colors = self._resolve_render_sector_polygons()

        self._journal_training_scenario("consumed")
        return observations, {}

    def step(self, action_dict: dict[str, Any]):  # ty: ignore[invalid-method-override]
        """
        - Cache aircraft by callsign (ac_by_id)
        - Interpret actions → TimedCommandList, record action with pos and current step
        - Apply commands and advance simulator one tick, extend past tracks for active aircraft
        - Roll out remaining episode steps with advance_multiple_steps(remaining_steps)
        - Compute global_reward from prev_world → planning_estimate
        - Update prev_world = planning_estimate
        - Compute dones: terminated if mission completed or missing, truncated if time limit hit;
        set terminated["__all__"] if no agents left or all within the configured success
        window (``success_time_dev_s``; see ``__init__`` for its defaults),
        else set truncated["__all__"] on time limit
        - Rewards: distribute each step reward across active agents with
        a global component plus acting-agent component
        - Observations: if global done/truncated, return final obs for enders (or all as needed);
        else pick next agent via _get_next_agents_rr (prioritizes time deviation),
        include enders' last obs, then prune finished agents from self.agents
        - Return (observations, rewards, terminated, truncated, infos)
        """
        assert self.prev_world is not None, "Previous world is not initialized"
        assert self.simulator.world is not None, "Simulator world is not initialized"

        present = {aircraft.call_sign for aircraft in self.simulator.world.aircraft}
        if any(target.callsign not in present for target in self.arrival_targets):
            from atc_env.completed_scenario import UnsupportedScenarioInput, capture_environment

            capture_environment(
                self, complete=False, error="unsupported_input:missing_retained_actor"
            )
            raise UnsupportedScenarioInput(
                "missing retained actor: cannot build truthful final observations"
            )

        # Invalidate per-step caches so they're recomputed lazily.
        self._conflict_override_cache = None
        self._step_deviation_proxies = None
        self._step_selection_infr_map = None
        self._step_prev_world_all_infr = None

        # Clear last_end_reason only after the first step completes.
        # This allows on_episode_end callbacks (called AFTER reset AND the first step
        # of the new episode) to still read the end reason from the previous episode.
        if self.step_counter > 0:
            self.last_end_reason = None

        # 1) Interpret actions
        action_time_s = float(self.simulator.world.age_in_seconds)
        interpreted_actions, acting_agents, action_nums, post_action_state = (
            self._interpret_actions(action_dict)
        )
        self.last_step_action_nums = dict(action_nums)
        conflict_noops = self._conflict_noops(action_nums)
        action_issue_positions = capture_aircraft_positions(self.simulator.world, acting_agents)

        # 2) Simulate and compute reward
        reward, did_act = self._simulate_and_reward(interpreted_actions)
        reward.action += compute_action_damping_penalty(
            action_nums,
            self.last_non_zero_action_time_s,
            action_time_s,
            options=self.reward_settings,
            time_between_actions=self.settings.env.time_between_actions,
        )
        reward.action += compute_speed_damping_penalty(
            action_nums,
            self.last_speed_action_time_s,
            action_time_s,
            options=self.reward_settings,
            time_between_actions=self.settings.env.time_between_actions,
        )
        self._update_last_non_zero_action_times(action_nums, action_time_s)
        self._update_last_speed_action_times(action_nums, action_time_s)
        self._update_recent_noop_holdoffs(action_nums)
        step_reward_total = float(reward.total())

        # 2b) Check if pending virtual turn waypoints were consumed during this advance
        self._check_pending_returns()

        # 3) Record action markers at command-issue positions
        self._record_actions_at_issue_positions(
            action_nums,
            acting_agents,
            action_issue_positions,
            step_reward_total,
            post_action_state,
            action_time_s,
        )

        # 4) Dones and truncations
        terminated, truncated, just_finished, reason = self._compute_dones()

        terminal_scale = reward_count_scale(
            len(self.arrival_targets), self.reward_settings["count_baseline"]
        )
        if reason == vocabulary.SUCCESS:
            reward.bonus += (
                self.reward_settings["success_bonus"] + float(self.last_success_margin_bonus)
            ) * terminal_scale
        elif reason == vocabulary.INFRINGEMENT and self.outcome_ledger.consume_failure():
            reward.bonus -= self.reward_settings["realized_failure_cost"] * terminal_scale
        elif reason == vocabulary.DELAYED:
            reward.bonus += self.reward_settings["delayed_clean_bonus"] * terminal_scale
        elif self._unresolved_failure(reason):
            reward.bonus -= self.reward_settings["unresolved_failure_cost"] * terminal_scale
        self.last_reward = reward
        self.cumulative_reward += float(reward.total())
        for component in self.episode_reward_components:
            self.episode_reward_components[component] += float(getattr(reward, component))

        # 5) Rewards: split into global + acting components
        rewards = distribute_step_reward(
            total_reward=float(reward.total()),
            acting_agents=set(acting_agents),
            active_agents=set(self.agents),
            global_share=float(self.reward_settings["global_share"]),
        )
        # A per-agent training signal; the team reward stays unchanged.
        for callsign in conflict_noops:
            rewards[callsign] -= self.reward_settings["conflict_noop_cost"]
        infos: dict[str, Any] = {}

        if terminated["__all__"] or truncated["__all__"]:
            from atc_env.completed_scenario import capture_environment

            capture_environment(self)

        # 6) Observations and agent list update
        enders = {a for a in self.agents if terminated[a] or truncated[a]}
        observations = self._prepare_observations_and_update_agents(
            terminated, truncated, just_finished, enders, did_act
        )
        self._close_untracked_actors(action_dict, terminated, truncated)

        return observations, rewards, terminated, truncated, infos
