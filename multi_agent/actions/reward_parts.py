"""Utilities for reward composition and rendering."""

from dataclasses import dataclass


@dataclass
class RewardParts:
    """Named components contributing to the step reward.

    Fields:
    - deviation: time deviation improvement component (positive is good)
    - conflict: conflict reduction component (positive is good)
    - action: action cost component (typically non-positive; negative is penalty)
    """

    deviation: float
    conflict: float
    action: float
    bonus: float = 0.0

    def total(self) -> float:
        return float(self.deviation + self.conflict + self.action + self.bonus)


@dataclass(frozen=True)
class ActedAircraftRewardRow:
    """Compact render row for one aircraft that acted in the last step."""

    agent_id: str
    action_num: int


def build_acted_aircraft_reward_rows(
    step_action_nums: dict[str, int],
) -> list[ActedAircraftRewardRow]:
    """Build deterministic acted-aircraft rows from last-step action/reward maps."""

    rows: list[ActedAircraftRewardRow] = []
    for agent_id in sorted(step_action_nums):
        rows.append(
            ActedAircraftRewardRow(
                agent_id=str(agent_id),
                action_num=int(step_action_nums[agent_id]),
            )
        )
    return rows
