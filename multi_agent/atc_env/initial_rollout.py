"""InitialRolloutMixin for ``ATCEnv``."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from actions.actions import (
    Action,
)
from actions.auto_solve import apply_auto_solve_to_routes, select_auto_solve_actions
from actions.rewards import (
    SHAPED_REWARD_MIN_BASELINE,
    compute_weighted_infringement_total,
    reward_count_scale,
)
from actions.timing_credit import TimingCreditState
from atc_env.coverage_consumers import (
    proxy_map,
    timing_map,
    timing_report,
)
from atc_env.environment_helpers import (
    capture_aircraft_positions,
)
from outcomes.infringement import (
    get_world_infringements,
)

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv


@dataclass(frozen=True)
class InitialPlanningMetrics:
    """Scalar metrics measured from a rollout-to-end planning world."""

    deviations_by_aircraft: dict[str, float]
    deviation_total: float | None
    deviation_average: float | None
    infringement_total: float
    numerical_timing: dict


class InitialRolloutMixin:
    def _prepare_initial_rollout(self: "ATCEnv") -> Any:
        """Prepare raw and policy-start baselines, then return the policy-start planning world."""

        assert self.simulator.world is not None, "Simulator world is not initialized"
        world = self.simulator.world
        self.episode_time_horizon = max(
            (float(target.eta_s) for target in world.arrival_targets),
            default=self.settings.env.episode_steps * self.settings.env.time_between_actions,
        )
        baseline_world = self._forecast(world, self.max_steps_per_episode)
        raw_metrics = self._measure_initial_planning_metrics(baseline_world)
        policy_start_world = baseline_world
        policy_start_metrics = raw_metrics

        target_window_s = float(self.settings.env.auto_solve_target_window_s)
        early_eligible_aircraft: list[Any] = []
        candidate_baseline_deviations: dict[str, float] = {}
        for ac in world.aircraft:
            if not self._route_trombone_store.has_excluded_section(ac.call_sign):
                continue
            evidence = timing_map(self.forecast_coverage).get(ac.call_sign)
            if evidence is None or evidence.upper_error_s >= -target_window_s:
                continue
            deviation_s = raw_metrics.deviations_by_aircraft.get(ac.call_sign)
            if deviation_s is None:
                continue
            early_eligible_aircraft.append(ac)
            candidate_baseline_deviations[ac.call_sign] = deviation_s

        if early_eligible_aircraft:
            routes_by_aircraft = {
                ac.call_sign: list(ac.route) if ac.route else [] for ac in early_eligible_aircraft
            }
            auto_actions = select_auto_solve_actions(
                world=world,
                baseline_deviations=candidate_baseline_deviations,
                route_store=self._route_trombone_store,
                routes_by_aircraft=routes_by_aircraft,
                rollout_steps=self.max_steps_per_episode,
                target_window_s=target_window_s,
                outcome_owner=self.outcome_state,
                time_between_actions=self.settings.env.time_between_actions,
            )
            if auto_actions:
                action_nums = self._resolve_auto_solve_action_nums(auto_actions)
                acting_agents = sorted(action_nums)
                action_issue_positions = capture_aircraft_positions(world, acting_agents)
                decision_time_s = float(world.age_in_seconds)
                commands, post_action_state = apply_auto_solve_to_routes(
                    actions=auto_actions,
                    route_store=self._route_trombone_store,
                    routes_by_aircraft=routes_by_aircraft,
                )
                self.simulator.world = self.outcome_state.advance_next(world, 0, commands)
                self._record_actions_at_issue_positions(
                    action_nums,
                    acting_agents,
                    action_issue_positions,
                    0.0,
                    post_action_state,
                    decision_time_s,
                    phase="pre_episode",
                )
                assert self.simulator.world is not None, "Simulator world is not initialized"
                policy_start_world = self._forecast(
                    self.simulator.world, self.max_steps_per_episode
                )
                policy_start_metrics = self._measure_initial_planning_metrics(policy_start_world)

        self._apply_initial_metric_baselines(raw_metrics, policy_start_metrics)
        count_scale = reward_count_scale(
            len(self.arrival_targets), self.reward_settings["count_baseline"]
        )
        self.timing_credit = TimingCreditState(
            [target.callsign for target in self.arrival_targets],
            timing_map(self.forecast_coverage),
            scaled_min_baseline=SHAPED_REWARD_MIN_BASELINE * count_scale,
            delay_max=float(self.reward_settings["max_for_zero_delay"]) * count_scale,
        )
        return policy_start_world

    def _measure_initial_planning_metrics(
        self: "ATCEnv", planning_world: Any
    ) -> InitialPlanningMetrics:
        deviation_proxies = proxy_map(self.forecast_coverage)
        report = timing_report(self.forecast_coverage)
        deviation_total = report["exact_total_absolute_error_s"]
        return InitialPlanningMetrics(
            deviations_by_aircraft=deviation_proxies,
            deviation_total=deviation_total,
            deviation_average=report["exact_mean_absolute_error_s"],
            numerical_timing=report,
            infringement_total=compute_weighted_infringement_total(
                get_world_infringements(planning_world, self.settings.core.safety),
                self.episode_time_horizon / 3.0,
                planning_world.age_in_seconds,
                gate_center=self.settings.core.reward.reward_infringement_gate_center,
            ),
        )

    def _apply_initial_metric_baselines(
        self: "ATCEnv",
        raw_metrics: InitialPlanningMetrics,
        policy_start_metrics: InitialPlanningMetrics,
    ) -> None:
        """Expose raw-scenario metrics separately from the policy-start baselines."""

        self.raw_initial_timing_report = raw_metrics.numerical_timing
        self.initial_timing_report = policy_start_metrics.numerical_timing
        self.raw_initial_deviation = raw_metrics.deviation_total
        self.raw_initial_deviation_avg = raw_metrics.deviation_average
        self.raw_initial_infringements_total = raw_metrics.infringement_total
        self.initial_deviation = policy_start_metrics.deviation_total
        self.initial_infringements_total = policy_start_metrics.infringement_total

    @staticmethod
    def _resolve_auto_solve_action_nums(auto_actions: dict[str, int]) -> dict[str, int]:
        return {callsign: int(Action.LENGTHEN_TROMBONE) for callsign in auto_actions}
