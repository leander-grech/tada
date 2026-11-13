"""ActionExecutionMixin for ``ATCEnv``."""

from typing import TYPE_CHECKING, Any

from actions.actions import (
    SPEED_ACTIONS,
    TURN_ACTION_SPECS,
    Action,
    ActionRecord,
    PostActionState,
    interpret_action,
)
from actions.reward_parts import RewardParts
from actions.rewards import (
    calculate_reward,
)
from atc_env.coverage_consumers import (
    timing_map,
)
from atc_env.environment_helpers import (
    target_agents_per_step,
)
from geometry.vectoring import (
    VIRTUAL_TURN_WAYPOINT_PREFIX,
    leading_virtual_turn_count,
)
from outcomes.infringement import (
    filter_infringements_by_window,
    get_world_infringements,
)
from simulator.commands import TimedCommandList

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv


class ActionExecutionMixin:
    def _interpret_actions(
        self: "ATCEnv", action_dict: dict[str, Any]
    ) -> tuple[TimedCommandList, list[str], dict[str, int], dict[str, PostActionState]]:
        """Translate actions to commands and return acting agents and action numbers"""

        interpreted_actions = TimedCommandList(commands=[])
        acting_agents: list[str] = []
        action_nums: dict[str, int] = {}
        action_state_dict: dict[str, PostActionState] = {}

        target_count = target_agents_per_step(self.settings.core.selection)
        allowed_to_act = getattr(self, "_agents_allowed_to_act", None)
        if allowed_to_act is None:
            eligible_callsigns = list(self.agents)
        else:
            eligible_callsigns = [
                callsign for callsign in self.agents if callsign in allowed_to_act
            ]
        selected_callsigns = [
            callsign for callsign in eligible_callsigns if callsign in action_dict
        ][:target_count]

        for callsign in selected_callsigns:
            action = action_dict[callsign]
            acting_agents.append(callsign)

            # Normalize once and store for later use (e.g., rendering)
            normalized = int(action) if not isinstance(action, Action) else int(action.value)
            action_nums[callsign] = normalized

            interpreted_action, post_action_state = interpret_action(
                action,
                callsign,
                self.simulator,
                self._route_trombone_store,
                self.settings.core.action,
            )
            action_state_dict[callsign] = post_action_state
            interpreted_actions.commands.extend(interpreted_action.commands)

        return interpreted_actions, acting_agents, action_nums, action_state_dict

    def _conflict_noops(self: "ATCEnv", action_nums: dict[str, int]) -> list[str]:
        """Acting agents charged ``reward.conflict_noop_cost``: they chose no-op although another
        action was allowed and the decision-time planning forecast predicts a conflict of at least
        the near-conflict severity within the neighbor window. Reads ``prev_world`` before the step
        replaces it and bypasses the per-step infringement caches, which selection fills later."""
        if not self.reward_settings["conflict_noop_cost"]:
            return []
        assert self.simulator.world is not None
        now = float(self.simulator.world.age_in_seconds)
        near = filter_infringements_by_window(
            get_world_infringements(self.prev_world, self.settings.core.safety),
            now,
            now + float(self.settings.env.neighbor_infringement_window_s),
        )
        threshold = float(self.settings.env.near_conflict_threshold)
        return [
            callsign
            for callsign, action_num in action_nums.items()
            if action_num == int(Action.DO_NOTHING)
            and self._last_action_mask_counts.get(callsign, 0) > 1
            and any(item.value >= threshold for item in near.get(callsign, ()))
        ]

    def _simulate_and_reward(
        self: "ATCEnv",
        interpreted_actions: TimedCommandList,
    ) -> tuple[RewardParts, bool]:
        """
        Apply actions, advance sim, update tracks, compute reward, and update prev_world.
        """

        assert self.simulator.world is not None, "Simulator world is not initialized"
        assert self.prev_world is not None, "Previous world is not initialized"

        # Advance simulation
        did_act = any(cmd.command is not None for cmd in interpreted_actions.commands)
        self.simulator.apply_action_and_advance(
            interpreted_actions,
            time_between_actions=self.settings.env.time_between_actions,
        )
        self.step_counter += 1

        # Update tracks
        for ac in self.simulator.world.aircraft:
            if not ac.mission.completed:
                pos = ac.state.position_nm_nm_ft
                self._past_tracks.setdefault(ac.call_sign, []).append((float(pos.x), float(pos.y)))

        # Roll out to episode end (or the minimum lookahead) from current simulator state.
        planning_estimate = self._forecast(self.simulator.world, self._planning_steps())
        previous_planning_world = self.prev_world
        # Pre-compute unfiltered infringements for the new planning world once.
        # Reused by calculate_reward and cached for selection/dones/observations.
        planning_infr = get_world_infringements(planning_estimate, self.settings.core.safety)
        reward = calculate_reward(
            interpreted_actions,
            previous_planning_world,
            planning_estimate,
            self.initial_deviation,
            self.initial_infringements_total,
            self.episode_time_horizon,
            float(self.simulator.world.age_in_seconds),
            len(self.arrival_targets),
            current_estimation_infr=planning_infr,
            timing_credit=self.timing_credit.update(timing_map(self.forecast_coverage)),
            options=self.reward_settings,
            gate_center=self.settings.core.reward.reward_infringement_gate_center,
            safety=self.settings.core.safety,
        )
        self.prev_world = planning_estimate
        self._step_prev_world_all_infr = planning_infr

        return reward, did_act

    def _record_actions_at_issue_positions(
        self: "ATCEnv",
        action_nums: dict[str, int],
        acting_agents: list[str],
        action_issue_positions: dict[str, tuple[float, float]],
        step_reward_total: float,
        post_action_state: dict[str, PostActionState] | None = None,
        decision_time_s: float | None = None,
        phase: str | None = None,
    ) -> None:
        """Append action markers using positions captured at command issue time."""
        if not acting_agents:
            return

        assert self.simulator.world is not None

        for callsign in acting_agents:
            xy_at_issue = action_issue_positions.get(callsign)
            if xy_at_issue is None:
                continue

            x_issue, y_issue = xy_at_issue
            act_num = int(action_nums.get(callsign, 0))
            record: ActionRecord = {
                "call_sign": callsign,
                "action": act_num,
                "x": float(x_issue),
                "y": float(y_issue),
                "step": int(self.step_counter),
                "reward": float(step_reward_total),
            }
            if decision_time_s is not None:
                record["decision_time_s"] = decision_time_s
            if phase is not None:
                record["phase"] = phase
            # Merge post-action state data when available
            if post_action_state and callsign in post_action_state:
                record.update(post_action_state[callsign])
            self._action_history.append(record)

            # Register pending return-to-route for turn actions
            if act_num in TURN_ACTION_SPECS:
                self._register_pending_return(callsign, record.get("route_after", []))

    def _register_pending_return(self: "ATCEnv", callsign: str, route_after: list[str]) -> None:
        """Queue a return-to-route entry for a single turn action."""
        return_wp = next(
            (wp for wp in route_after if not wp.startswith(VIRTUAL_TURN_WAYPOINT_PREFIX)),
            None,
        )
        if return_wp is not None:
            self._pending_returns[callsign].append(return_wp)

    def _check_pending_returns(self: "ATCEnv") -> None:
        """Emit RETURN_TO_ROUTE records when virtual turn waypoints are consumed.

        Each pending return corresponds to one VT waypoint in the route. When the
        aircraft passes a VT (simulator removes it), len(pending) > vt_count and
        we emit the corresponding RETURN_TO_ROUTE.
        """
        if not self._pending_returns:
            return
        assert self.simulator.world is not None
        current_time_s = float(self.simulator.world.age_in_seconds)

        for callsign in list(self._pending_returns):
            pending = self._pending_returns[callsign]
            if not pending:
                self._pending_returns.pop(callsign)
                continue

            ac = self.simulator.world.get_aircraft_by_callsign(callsign)
            if ac is None:
                self._pending_returns.pop(callsign)
                continue

            route = ac.route or []
            vt_count = leading_virtual_turn_count(route)
            to_emit = len(pending) - vt_count

            for _ in range(to_emit):
                return_wp = pending.popleft()
                pos = ac.state.position_nm_nm_ft
                self._action_history.append(
                    {
                        "call_sign": callsign,
                        "action": int(Action.RETURN_TO_ROUTE),
                        "x": float(pos.x),
                        "y": float(pos.y),
                        "step": int(self.step_counter),
                        "reward": 0.0,
                        "decision_time_s": current_time_s,
                        "route_after": [rp.waypoint_name for rp in route],
                        "return_wp": return_wp,
                    }
                )

            if not pending:
                self._pending_returns.pop(callsign)

    def _update_last_non_zero_action_times(
        self: "ATCEnv", action_nums: dict[str, int], action_time_s: float
    ) -> None:
        """Track the latest timestamp when each agent took a non-speed, non-idle action."""
        for callsign, action_num in action_nums.items():
            if callsign not in self.agents:
                continue
            if action_num != Action.DO_NOTHING and action_num not in SPEED_ACTIONS:
                self.last_non_zero_action_time_s[callsign] = float(action_time_s)

    def _update_last_speed_action_times(
        self: "ATCEnv", action_nums: dict[str, int], action_time_s: float
    ) -> None:
        """Track the latest timestamp when each agent took a speed action."""
        for callsign, action_num in action_nums.items():
            if callsign not in self.agents:
                continue
            if action_num in SPEED_ACTIONS:
                self.last_speed_action_time_s[callsign] = float(action_time_s)
