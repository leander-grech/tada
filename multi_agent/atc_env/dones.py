"""DonesMixin for ``ATCEnv``."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from atc_env.coverage_consumers import (
    proxy_map,
    success_margin,
)
from outcomes import vocabulary

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv


class DonesMixin:
    def _compute_success_margin(
        self: "ATCEnv",
        *,
        deviation_proxies: dict[str, float],
        arrival_target_callsigns: set[str],
    ) -> float:
        """Compute normalized success margin in [0, 1] above terminal thresholds.

        Scaled by the bound training window, never by an
        evaluation-only ``success_time_dev_s``, so this reward bonus keeps the
        magnitude it had during training.
        """
        payload = self._training_payload
        threshold = (
            payload["thresholds"]["training_success_s"]
            if payload is not None
            else self.settings.env.satisfactory_time_dev
        )
        return success_margin(deviation_proxies, arrival_target_callsigns, float(threshold))

    def _compute_dones(
        self: "ATCEnv",
    ) -> tuple[dict[str, bool], dict[str, bool], list[str], str | None]:
        """Compute per-agent termination/truncation and set __all__ flags."""

        assert self.prev_world is not None and self.simulator.world is not None, (
            "World states are not initialized"
        )

        terminated: dict[str, bool] = dict.fromkeys(self.agents, False)
        truncated: dict[str, bool] = dict.fromkeys(self.agents, False)

        just_finished: list[str] = []
        aircraft_by_callsign = {ac.call_sign: ac for ac in self.simulator.world.aircraft}
        completed_callsigns = {
            evidence.callsign
            for evidence in self.outcome_ledger.snapshot().arrivals
            if evidence.completed
        }
        for callsign in self.agents:
            ac = aircraft_by_callsign.get(callsign)
            if callsign in completed_callsigns or (ac is not None and ac.mission.completed):
                terminated[callsign] = True
                just_finished.append(callsign)

        hit_time_limit = self.step_counter >= self.policy_time_limit_steps
        self.failed_target_callsigns = tuple(
            target.callsign
            for target in self.arrival_targets
            if (aircraft := aircraft_by_callsign.get(target.callsign)) is not None
            and aircraft.mission.failed
        )
        outcome = self.outcome_state.classify(
            self.simulator.world,
            threshold_s=self.success_time_dev_s,
            policy_limit_reached=hit_time_limit and not self.failed_target_callsigns,
            verification_cap=self.noop_verification_max_seconds,
        )
        physical = self.outcome_state.physically_complete or bool(self.failed_target_callsigns)
        if self.failed_target_callsigns and outcome.kind != vocabulary.INFRINGEMENT:
            outcome = replace(
                outcome,
                kind=vocabulary.UNRESOLVED,
                reasons=outcome.reasons + ("native_mission_failed",),
            )
            self.outcome_state.last_outcome = outcome
        reason = None
        if outcome.kind == vocabulary.INFRINGEMENT:
            reason = vocabulary.INFRINGEMENT
        elif outcome.kind in vocabulary.SUCCESS_KINDS:
            reason = vocabulary.SUCCESS
        elif physical:
            reason = (
                vocabulary.DELAYED
                if outcome.kind == vocabulary.SAFE_DELAYED
                else vocabulary.UNRESOLVED
            )
        elif hit_time_limit:
            reason = vocabulary.TIMEOUT
        is_terminal = reason is not None and (physical or outcome.kind == vocabulary.INFRINGEMENT)
        is_truncated = reason is not None and not is_terminal
        for callsign in self.agents:
            if is_terminal:
                terminated[callsign] = True
            elif is_truncated and not terminated[callsign]:
                truncated[callsign] = True
        terminated["__all__"] = is_terminal
        truncated["__all__"] = is_truncated
        self.last_success_margin_bonus = 0.0
        if reason == vocabulary.SUCCESS:
            evidence = (
                self.last_verification.outcome
                if self.last_verification is not None
                else self.outcome_ledger.snapshot()
            )
            self.last_success_margin_bonus = float(
                self.reward_settings["success_margin_bonus_max"]
            ) * self._compute_success_margin(
                deviation_proxies=proxy_map(evidence),
                arrival_target_callsigns={t.callsign for t in self.arrival_targets},
            )

        self.last_end_reason = reason
        if reason is not None:
            self._scenario_finished = True

        return terminated, truncated, just_finished, reason

    @staticmethod
    def _close_untracked_actors(
        action_dict: dict[str, Any],
        terminated: dict[str, bool],
        truncated: dict[str, bool],
    ) -> None:
        """Report a terminal flag for agents this env no longer tracks.

        This cleanup applies in both training and evaluation. RLlib ends a
        ``MultiAgentEpisode`` as soon as every agent it tracks is done, which
        happens mid-scenario here because aircraft are observed a few at a time
        and dropped when their mission completes. It then reads the next step's
        observations - a dropped aircraft's final observation among them - as a
        fresh episode's reset observations, so it keeps asking that aircraft to
        act after this env stopped listing it. Without a done flag RLlib holds
        the action for a matching observation that never comes and fails the
        episode's truncation with ``MultiAgentEnvError``. Terminating (never
        truncating) such an agent lets RLlib close it against its previous
        observation instead.
        """
        for agent_id in action_dict:
            if agent_id not in terminated:
                terminated[agent_id] = True
                truncated[agent_id] = False
