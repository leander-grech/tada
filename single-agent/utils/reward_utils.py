"""Reward component container."""

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
    goal: float = 0.0  # dense goal-closeness bonus (grows as predicted deviation -> 0)

    def total(self) -> float:
        return float(self.deviation + self.conflict + self.action + self.bonus + self.goal)
