"""AgentSelectionMixin for ``ATCEnv``."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from actions.actions import (
    Action,
)
from atc_env.coverage_consumers import (
    proxy_map,
)
from atc_env.environment_helpers import (
    StageCSelectionCandidate,
    StageCSoftSuppression,
    capture_aircraft_positions,
    compute_stage_c_priority_score,
    get_action_selection_mode,
    get_sector_selection_groups,
    select_stage_c_candidate,
    tiny_stable_noise,
)
from atc_env.sector_selection import (
    SectorGroup,
    SelectedGroupIndex,
    assign_aircraft_to_selected_groups,
    get_selected_group_index_cached_from_normalized,
)
from outcomes.infringement import (
    filter_infringements_by_window,
    get_world_infringements,
)
from simulator.route_section_store import is_past_section_end

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv


@dataclass(frozen=True)
class SelectionNoopHoldoff:
    """Recent explicit do-nothing state used as a soft reselection suppressor."""

    remaining_steps: int
    baseline_score: float


class AgentSelectionMixin:
    def _get_next_agents_rr(
        self: "ATCEnv",
        agents_to_choose_from: list[str] | None = None,
        did_act: bool | None = None,
    ) -> list[str]:
        """Dispatch to configured next-agent selection mode."""
        selection_mode = get_action_selection_mode(self.settings.core.selection)
        if selection_mode == "sector_based":
            return self._select_sector_based_next_agents(agents_to_choose_from)
        return self._select_single_next_agents(agents_to_choose_from, did_act)

    def _select_single_next_agents(
        self: "ATCEnv",
        agents_to_choose_from: list[str] | None = None,
        did_act: bool | None = None,
    ) -> list[str]:
        """Choose one next agent by (A) bounded repeat, then (B) priority score.

        A) If the last agent actually issued a non-None command in the previous step,
        allow a bounded repeat up to MAX_CONSECUTIVE_ACTS.
        B) Otherwise, pick the argmax of:
        w_conflict * max_predicted_severity + w_dev * normalized(|deviation|),
        with a tiny stable tie-breaker for determinism.
        Conflict severity is already weighted into the score, so aircraft in
        conflict naturally rank higher without needing a cooldown-bypassing override.
        """
        assert self.simulator.world is not None
        assert self.prev_world is not None

        # --- Candidate set -----------------------------------------------------
        agents = self._resolve_controlled_candidates(agents_to_choose_from)
        if not agents:
            self._set_selection_observability(controlled_count=0, outside_excluded_count=0)
            return []
        self._set_selection_observability(controlled_count=len(agents), outside_excluded_count=0)

        # Bookkeeping defaults (keep dict compact by tracking current candidates only).
        max_consec = int(self.settings.env.max_consecutive_acts)
        self.consecutive_acts = {a: self.consecutive_acts.get(a, 0) for a in agents}
        self._normalize_selection_suppressors_for_candidates(agents)

        # --- (A) bounded repeat of last agent -----------------------------------
        # If last agent just acted (produced a non-None command) and streak not maxed, repeat it.
        capped_repeat_agent: str | None = None
        if (
            self.last_acted_agent in agents
            and did_act is True
            and (self.last_acted_agent is not None)
        ):
            if self.consecutive_acts.get(self.last_acted_agent, 0) < max_consec:
                a = self.last_acted_agent
                self._decrement_selection_suppressors()
                self.consecutive_acts[a] = self.consecutive_acts.get(a, 0) + 1
                return [a]
            capped_repeat_agent = self.last_acted_agent

        selection_agents = agents
        if capped_repeat_agent is not None:
            uncapped_agents = [agent_id for agent_id in agents if agent_id != capped_repeat_agent]
            if uncapped_agents:
                selection_agents = uncapped_agents

        # If we didn't repeat, start a new streak for whoever we pick next.
        for k in list(self.consecutive_acts.keys()):
            self.consecutive_acts[k] = 0

        max_conf, dev_abs, dev_signed = self._build_sector_selection_metrics(agents)

        # --- priority score selection ------------------------------------------
        chosen = self._select_stage_c_candidate(
            selection_agents,
            max_conf=max_conf,
            dev_abs=dev_abs,
            dev_signed=dev_signed,
        )
        self._decrement_selection_suppressors()
        self._apply_stage_cooldown([chosen])
        self.last_acted_agent = chosen
        self.consecutive_acts = {chosen: 1}
        return [chosen]

    def _select_sector_based_next_agents(
        self: "ATCEnv",
        agents_to_choose_from: list[str] | None = None,
    ) -> list[str]:
        """Choose up to one agent per selected sector group using sector membership."""
        agents = self._resolve_controlled_candidates(agents_to_choose_from)
        if not agents:
            self._set_selection_observability(controlled_count=0, outside_excluded_count=0)
            return []

        self._normalize_selection_suppressors_for_candidates(agents)
        selected_groups = self._resolve_selected_group_index().selected_groups
        candidates_by_group = self._group_sector_candidates(agents)
        if not candidates_by_group:
            return self._finalize_sector_based_selection(
                chosen_agents=[],
                controlled_count=len(agents),
                outside_excluded_count=len(agents),
            )

        max_conf, dev_abs, dev_signed = self._build_sector_selection_metrics(agents)
        grouped_union = {
            agent_id
            for grouped_callsigns in candidates_by_group.values()
            for agent_id in grouped_callsigns
        }
        outside_candidates = [agent_id for agent_id in agents if agent_id not in grouped_union]

        chosen_agents: list[str] = []
        for group_name in selected_groups:
            group_candidates = candidates_by_group.get(group_name, [])
            if not group_candidates:
                continue
            chosen_agents.append(
                self._select_stage_c_candidate(
                    group_candidates,
                    max_conf=max_conf,
                    dev_abs=dev_abs,
                    dev_signed=dev_signed,
                )
            )

        return self._finalize_sector_based_selection(
            chosen_agents=chosen_agents,
            controlled_count=len(agents),
            outside_excluded_count=len(outside_candidates),
        )

    def _finalize_sector_based_selection(
        self: "ATCEnv",
        *,
        chosen_agents: list[str],
        controlled_count: int,
        outside_excluded_count: int,
    ) -> list[str]:
        self._set_selection_observability(
            controlled_count=controlled_count,
            outside_excluded_count=outside_excluded_count,
        )
        self._decrement_selection_suppressors()
        self._apply_stage_cooldown(chosen_agents)
        self.last_acted_agent = None
        self.consecutive_acts = {}
        return chosen_agents

    def _resolve_controlled_candidates(
        self: "ATCEnv", agents_to_choose_from: list[str] | None
    ) -> list[str]:
        assert self.simulator.world is not None
        assert self.prev_world is not None
        agents = self.agents if agents_to_choose_from is None else agents_to_choose_from
        controlled = [
            agent_id
            for agent_id in agents
            if self.simulator.world.is_under_control(agent_id)
            and not is_past_section_end(
                self.simulator.world.get_aircraft_by_callsign(agent_id).route or []
            )
        ]

        # Update lockouts and apply filtering (cache time_devs for reuse)
        if self._step_deviation_proxies is None:
            self._step_deviation_proxies = proxy_map(self.forecast_coverage)
        self._update_lockouts(self._step_deviation_proxies)
        conflict_overrides = self._get_conflict_override_agents()
        return [
            a for a in controlled if a not in self._locked_out_agents or a in conflict_overrides
        ]

    def _update_lockouts(self: "ATCEnv", deviation_proxies: dict[str, float]) -> None:
        """Update lockout set from current deviation state for controlled aircraft only."""
        assert self.simulator.world is not None
        threshold = float(self.lockout_threshold_s)
        if threshold <= 0.0:
            self._locked_out_agents = set()
            return

        self._locked_out_agents = {
            agent_id
            for agent_id in self.agents
            if self.simulator.world.is_under_control(agent_id)
            and agent_id in deviation_proxies
            and abs(float(deviation_proxies[agent_id])) < threshold
        }

    def _get_conflict_override_agents(self: "ATCEnv") -> set[str]:
        """Return locked-out agents that must be temporarily re-enabled for selection.

        Two locked-out aircraft with a mutual predicted conflict (severity > 0)
        are both re-enabled so the agent can resolve the conflict.
        """
        if self._conflict_override_cache is not None:
            return self._conflict_override_cache

        if not self._locked_out_agents:
            self._conflict_override_cache = set()
            return self._conflict_override_cache

        assert self.prev_world is not None
        assert self.simulator.world is not None
        infr_map = self._get_selection_infr_map()

        near_conflict = float(self.settings.env.near_conflict_threshold)
        overrides: set[str] = set()
        for agent_id in self._locked_out_agents:
            for infr in infr_map.get(agent_id, ()):
                if infr.value > 0.0 and infr.counterpart in self._locked_out_agents:
                    # Mutual locked pair: any conflict re-enables both
                    overrides.add(agent_id)
                    overrides.add(infr.counterpart)
                elif infr.value >= near_conflict:
                    # Locked vs unlocked: re-enable locked agent at significant severity
                    overrides.add(agent_id)

        self._conflict_override_cache = overrides
        return overrides

    def _group_sector_candidates(self: "ATCEnv", agents: list[str]) -> dict[SectorGroup, list[str]]:
        selected_index = self._resolve_selected_group_index()
        grouped = assign_aircraft_to_selected_groups(
            capture_aircraft_positions(self.simulator.world, agents),
            selected_index,
        )
        return grouped

    def _resolve_selected_group_index(self: "ATCEnv") -> SelectedGroupIndex:
        selected_index = getattr(self, "_selected_group_index", None)
        if selected_index is not None:
            return selected_index
        selected_index = get_selected_group_index_cached_from_normalized(
            get_sector_selection_groups(self.settings.core.selection)
        )
        self._selected_group_index = selected_index
        return selected_index

    def _get_prev_world_all_infr(self: "ATCEnv") -> dict[str, list]:
        """Return unfiltered infringements for prev_world, cached per step."""
        if self._step_prev_world_all_infr is None:
            self._step_prev_world_all_infr = get_world_infringements(
                self.prev_world, self.settings.core.safety
            )
        return self._step_prev_world_all_infr

    def _get_selection_infr_map(self: "ATCEnv") -> dict[str, list]:
        """Return the selection-phase infringement map, cached per step."""
        if self._step_selection_infr_map is None:
            assert self.simulator.world is not None
            current_time_s = float(self.simulator.world.age_in_seconds)
            lookahead_s = int(self.settings.env.neighbor_infringement_window_s)
            self._step_selection_infr_map = filter_infringements_by_window(
                self._get_prev_world_all_infr(),
                current_time_s,
                current_time_s + float(lookahead_s),
            )
        return self._step_selection_infr_map

    def _build_sector_selection_metrics(
        self: "ATCEnv",
        agents: list[str],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        assert self.prev_world is not None
        infr_map = self._get_selection_infr_map()
        if self._step_deviation_proxies is None:
            self._step_deviation_proxies = proxy_map(self.forecast_coverage)
        tdev = self._step_deviation_proxies
        max_conf = {a: max((x.value for x in infr_map.get(a, ())), default=0.0) for a in agents}
        dev_abs = {a: abs(float(tdev[a])) for a in agents if a in tdev}
        dev_signed = {a: float(tdev[a]) for a in agents if a in tdev}
        return max_conf, dev_abs, dev_signed

    def _normalize_selection_suppressors_for_candidates(self: "ATCEnv", agents: list[str]) -> None:
        self.stage_cooldown_by_agent = {
            agent_id: int(self.stage_cooldown_by_agent.get(agent_id, 0))
            for agent_id in agents
            if int(self.stage_cooldown_by_agent.get(agent_id, 0)) > 0
        }
        self.recent_noop_holdoff_by_agent = {
            agent_id: holdoff
            for agent_id, holdoff in self.recent_noop_holdoff_by_agent.items()
            if agent_id in agents and holdoff.remaining_steps > 0
        }

    def _select_stage_c_candidate(
        self: "ATCEnv",
        candidates: list[str],
        *,
        max_conf: dict[str, float],
        dev_abs: dict[str, float],
        dev_signed: dict[str, float],
    ) -> str:
        scores = self._build_stage_c_scores(
            candidates,
            max_conf=max_conf,
            dev_abs=dev_abs,
            dev_signed=dev_signed,
        )
        selection_candidates = [
            StageCSelectionCandidate(
                agent_id=agent_id,
                score=scores[agent_id],
                suppression=self._build_stage_c_soft_suppression(
                    agent_id,
                    score=scores[agent_id],
                ),
            )
            for agent_id in candidates
        ]
        return select_stage_c_candidate(
            selection_candidates,
            self.settings.core.selection.selection_soft_suppression_dominance_margin,
        )

    def _build_stage_c_scores(
        self: "ATCEnv",
        candidates: list[str],
        *,
        max_conf: dict[str, float],
        dev_abs: dict[str, float],
        dev_signed: dict[str, float],
    ) -> dict[str, float]:
        return {
            agent_id: self._stage_c_score(
                agent_id,
                max_conf=max_conf,
                dev_abs=dev_abs,
                dev_signed=dev_signed,
            )
            for agent_id in candidates
        }

    def _build_stage_c_soft_suppression(
        self: "ATCEnv",
        agent_id: str,
        *,
        score: float,
    ) -> StageCSoftSuppression:
        holdoff = self.recent_noop_holdoff_by_agent.get(agent_id)
        recent_noop_suppressed = False
        if holdoff is not None and holdoff.remaining_steps > 0:
            release_margin = max(
                0.0, float(self.settings.env.selection_soft_suppression_dominance_margin)
            )
            recent_noop_suppressed = score < holdoff.baseline_score + release_margin
        return StageCSoftSuppression(
            cooling_down=int(self.stage_cooldown_by_agent.get(agent_id, 0)) > 0,
            recent_noop=recent_noop_suppressed,
        )

    def _stage_c_score(
        self: "ATCEnv",
        agent_id: str,
        *,
        max_conf: dict[str, float],
        dev_abs: dict[str, float],
        dev_signed: dict[str, float],
    ) -> float:
        if agent_id not in dev_abs:
            # Unknown timing has no satisfaction discount or invented error sign.
            return float(self.settings.env.selection_weight_conflict) * max_conf[
                agent_id
            ] + 1e-6 * tiny_stable_noise(agent_id)
        return compute_stage_c_priority_score(
            max_conflict=float(max_conf[agent_id]),
            abs_deviation_s=float(dev_abs[agent_id]),
            deviation_s=float(dev_signed[agent_id]),
            settings=self.settings.core.selection,
        ) + 1e-6 * tiny_stable_noise(agent_id)

    def _decrement_selection_suppressors(self: "ATCEnv") -> None:
        for agent_id in list(self.stage_cooldown_by_agent.keys()):
            remaining = int(self.stage_cooldown_by_agent[agent_id]) - 1
            if remaining > 0:
                self.stage_cooldown_by_agent[agent_id] = remaining
            else:
                self.stage_cooldown_by_agent.pop(agent_id, None)
        for agent_id in list(self.recent_noop_holdoff_by_agent.keys()):
            holdoff = self.recent_noop_holdoff_by_agent[agent_id]
            remaining = holdoff.remaining_steps - 1
            if remaining > 0:
                self.recent_noop_holdoff_by_agent[agent_id] = SelectionNoopHoldoff(
                    remaining_steps=remaining,
                    baseline_score=holdoff.baseline_score,
                )
            else:
                self.recent_noop_holdoff_by_agent.pop(agent_id, None)

    def _apply_stage_cooldown(self: "ATCEnv", chosen_agents: list[str]) -> None:
        cooldown_steps = int(self.settings.env.selection_cooldown_steps)
        if cooldown_steps <= 0:
            return
        for agent_id in chosen_agents:
            self.stage_cooldown_by_agent[agent_id] = cooldown_steps

    def _update_recent_noop_holdoffs(self: "ATCEnv", action_nums: dict[str, int]) -> None:
        """Track explicit do-nothing actions as a soft reselection suppressor."""
        noop_holdoff_steps = int(self.settings.env.selection_recent_noop_holdoff_steps)
        if noop_holdoff_steps <= 0:
            self.recent_noop_holdoff_by_agent = {}
            return

        explicit_noops = [
            callsign
            for callsign, action_num in action_nums.items()
            if callsign in self.agents
            and int(action_num) == int(Action.DO_NOTHING)
            and int(self._last_action_mask_counts.get(callsign, 0)) > 1
        ]
        for callsign, action_num in action_nums.items():
            if callsign not in self.agents or int(action_num) == int(Action.DO_NOTHING):
                continue
            self.recent_noop_holdoff_by_agent.pop(callsign, None)

        if not explicit_noops:
            return

        max_conf, dev_abs, dev_signed = self._build_sector_selection_metrics(explicit_noops)
        scores = self._build_stage_c_scores(
            explicit_noops,
            max_conf=max_conf,
            dev_abs=dev_abs,
            dev_signed=dev_signed,
        )
        for callsign in explicit_noops:
            self.recent_noop_holdoff_by_agent[callsign] = SelectionNoopHoldoff(
                remaining_steps=noop_holdoff_steps,
                baseline_score=scores[callsign],
            )

    def _set_selection_observability(
        self: "ATCEnv", *, controlled_count: int, outside_excluded_count: int
    ) -> None:
        """Store per-step selector counters for callback-level observability metrics."""
        self.last_controlled_candidate_count = max(0, int(controlled_count))
        self.last_outside_excluded_count = max(0, int(outside_excluded_count))
