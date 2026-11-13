"""ObservationBuilderMixin for ``ATCEnv``."""

from typing import TYPE_CHECKING, Any

import numpy as np

from atc_env.coverage_consumers import (
    proxy_map,
    timing_map,
)
from config.observation_fields import GLOBAL_CONTEXT_FIELDS
from models.feature_transforms import scalar_clamp
from models.observations import make_agent_observations_from_simulator
from outcomes.infringement import (
    filter_infringements_by_window,
)

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv


class ObservationBuilderMixin:
    def _record_last_action_mask_counts(self: "ATCEnv", observations: dict[str, Any]) -> None:
        """Store per-agent action-mask counts from the latest emitted observations."""
        action_mask_counts = {
            agent_id: int(obs["action_mask"].sum())
            for agent_id, obs in observations.items()
            if isinstance(obs, dict) and "action_mask" in obs
        }
        if action_mask_counts:
            self._last_action_mask_counts = action_mask_counts

    def _prepare_observations_and_update_agents(
        self: "ATCEnv",
        terminated: dict[str, bool],
        truncated: dict[str, bool],
        just_finished: list[str],
        enders: Any,
        did_act: bool | None = None,
    ) -> dict[str, Any]:
        """Build observations for the next step and prune finished agents."""

        assert self.prev_world is not None, "Previous world is not initialized"

        is_terminal_step = bool(terminated["__all__"] or truncated["__all__"])
        # Observations
        if is_terminal_step:
            # Must still provide last obs for ended agents for bootstrapping
            observe_list = sorted(enders) if enders else list(self.agents)
            self._agents_allowed_to_act = set()
        else:
            finished = set(just_finished)
            remaining_agents = [a for a in self.agents if a not in finished]
            next_agents_to_act = self._get_next_agents_rr(remaining_agents, did_act)
            self._agents_allowed_to_act = set(next_agents_to_act)
            # Include enders' last obs as well (they won't act again, but RLlib needs the final obs)
            observe_list = (
                sorted(set(next_agents_to_act) | enders) if enders else list(next_agents_to_act)
            )

            # If no agents are under control but some are still flying (not yet in control zone),
            # provide an observation to prevent RLlib from ending the episode prematurely.
            # Action masking will handle restricting available actions.
            if not observe_list and remaining_agents:
                observe_list = [remaining_agents[0]]
        # Keep scheduler observability counters from the latest non-terminal step.
        # Terminal-step observe lists include bootstrap-only enders and would skew
        # blocked/observed diagnostics toward zero.
        if not is_terminal_step:
            self._record_observation_selection_metrics(
                observe_list,
                self._agents_allowed_to_act,
            )

        # Build observations before pruning to match RLlib expectations
        observations = self._build_observations(observe_list)
        self._record_last_action_mask_counts(observations)

        # Prune finished agents for next step bookkeeping
        if just_finished:
            finished = set(just_finished)
            self.agents = [a for a in self.agents if a not in finished]
            for agent_id in finished:
                self.last_non_zero_action_time_s.pop(agent_id, None)
                self.last_speed_action_time_s.pop(agent_id, None)
                self.stage_cooldown_by_agent.pop(agent_id, None)
                self.recent_noop_holdoff_by_agent.pop(agent_id, None)
                self.consecutive_acts.pop(agent_id, None)
                self._locked_out_agents.discard(agent_id)
                self._last_action_mask_counts.pop(agent_id, None)

        assert isinstance(observations, dict), "Observations is not a dict"
        return observations

    def _record_observation_selection_metrics(
        self: "ATCEnv",
        observe_list: list[str],
        allowed_to_act: set[str] | None,
    ) -> None:
        """Store observed-vs-eligible mismatch counters for scheduler diagnostics."""
        allowed = set(allowed_to_act or ())
        self.last_observation_count = len(observe_list)
        self.last_blocked_observation_count = sum(
            1 for callsign in observe_list if callsign not in allowed
        )

    def _build_global_context(
        self: "ATCEnv",
        *,
        forecast_infringements: dict[str, list[Any]],
        deviation_proxies: dict[str, float],
        current_time_s: float,
        effectively_locked_agents: set[str] | None = None,
    ) -> list[float]:
        """Build shared critic-side context from global traffic state.

        Feature order:
        active fraction, actable fraction, peak forecast severity, mean forecast
        peak severity, urgent fraction, mean absolute deviation, max absolute
        deviation, step fraction, deviation std, early-aircraft fraction,
        effectively locked fraction.
        """
        assert self.simulator.world is not None

        active_callsigns = sorted(
            aircraft.call_sign
            for aircraft in self.simulator.world.aircraft
            if not aircraft.mission.completed
        )
        if not active_callsigns:
            return [0.0] * len(GLOBAL_CONTEXT_FIELDS)

        active_count = float(len(active_callsigns))
        max_episode_aircraft = max(1, int(self.episode_max_aircraft_count))
        allowed_to_act = self._agents_allowed_to_act
        locked_out_agents = effectively_locked_agents
        if locked_out_agents is None:
            locked_out_agents = self._locked_out_agents
        near_conflict_threshold = float(self.settings.env.near_conflict_threshold)
        near_term_horizon_s = float(current_time_s) + float(
            self.settings.env.neighbor_infringement_window_s
        )
        encode_deviation = self.feature_settings.encode_deviation
        timing_evidence = timing_map(self.forecast_coverage)

        actable_count = 0
        locked_out_count = 0
        urgent_count = 0
        early_count = 0
        forecast_max_sum = 0.0
        forecast_max_peak = 0.0
        abs_deviations: list[float] = []

        for callsign in active_callsigns:
            if callsign in allowed_to_act:
                actable_count += 1
            if callsign in locked_out_agents:
                locked_out_count += 1

            if callsign in deviation_proxies:
                abs_deviations.append(abs(float(deviation_proxies[callsign])))
                evidence = timing_evidence.get(callsign)
                if evidence is not None and evidence.upper_error_s < 0:
                    early_count += 1

            max_severity = 0.0
            urgent = False
            for item in forecast_infringements.get(callsign, ()):
                severity = float(item.value)
                timestamp_s = float(item.timestamp)
                max_severity = max(max_severity, severity)
                if (
                    not urgent
                    and severity >= near_conflict_threshold
                    and timestamp_s <= near_term_horizon_s
                ):
                    urgent = True

            forecast_max_sum += max_severity
            forecast_max_peak = max(forecast_max_peak, max_severity)
            urgent_count += int(urgent)

        deviation_std = float(np.std(abs_deviations)) if abs_deviations else 0.0

        return [
            float(scalar_clamp(active_count / float(max_episode_aircraft), 0.0, 1.0)),
            float(
                scalar_clamp(
                    float(actable_count) / active_count,
                    0.0,
                    1.0,
                )
            ),
            float(forecast_max_peak),
            float(scalar_clamp(forecast_max_sum / active_count, 0.0, 1.0)),
            float(scalar_clamp(float(urgent_count) / active_count, 0.0, 1.0)),
            encode_deviation(sum(abs_deviations) / max(1, len(abs_deviations))),
            encode_deviation(max(abs_deviations, default=0.0)),
            float(
                scalar_clamp(
                    float(self.step_counter) / float(max(1, self.max_steps_per_episode)),
                    0.0,
                    1.0,
                )
            ),
            encode_deviation(deviation_std),
            float(scalar_clamp(float(early_count) / max(1, len(abs_deviations)), 0.0, 1.0)),
            float(scalar_clamp(float(locked_out_count) / active_count, 0.0, 1.0)),
            float(len(abs_deviations) / active_count),
        ]

    def _build_observations(self: "ATCEnv", observe_list: list[str]) -> dict[str, Any]:
        """Construct observations from current state + full-horizon deviation projection."""
        assert self.prev_world is not None and self.simulator.world is not None
        if self._step_deviation_proxies is None:
            self._step_deviation_proxies = proxy_map(self.forecast_coverage)
        deviation_proxies = self._step_deviation_proxies
        current_time_s = float(self.simulator.world.age_in_seconds)
        forecast_horizon_s = self._planning_steps() * int(self.settings.env.time_between_actions)
        # forecast_infr_tuples uses a broader window than _get_selection_infr_map()
        # (full remaining episode vs. fixed 270s) so the critic sees long-horizon severity.
        forecast_infr_tuples = filter_infringements_by_window(
            self._get_prev_world_all_infr(),
            current_time_s,
            current_time_s + float(forecast_horizon_s),
        )
        # Effectively locked = locked out minus conflict overrides (those get under_control=1)
        effectively_locked = self._locked_out_agents - self._get_conflict_override_agents()
        global_context = self._build_global_context(
            forecast_infringements=forecast_infr_tuples,
            deviation_proxies=deviation_proxies,
            current_time_s=current_time_s,
            effectively_locked_agents=effectively_locked,
        )
        observations = make_agent_observations_from_simulator(
            self.simulator,
            observe_list,
            current_infringements=self._get_selection_infr_map(),
            deviation_proxies=deviation_proxies,
            route_trombone_store=self._route_trombone_store,
            global_context=global_context,
            last_non_zero_action_times=self.last_non_zero_action_time_s,
            allowed_to_act=self._agents_allowed_to_act,
            locked_out_agents=effectively_locked,
            aircraft_airports=self._aircraft_airports,
            settings=self.feature_settings,
            scales=self.settings.core.observation,
            action_settings=self.settings.core.action,
        )
        return observations
