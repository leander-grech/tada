"""Bounded persistent timing credit over a frozen authoritative target population."""

import math
from collections.abc import Mapping, Sequence

from outcomes.evidence import TimingEvidence, forecast_deviation_proxy


class TimingCreditState:
    """Retain anchors through gaps; debit deterioration at reacquisition.

    For each known update delta <= anchor_before - anchor_after. Therefore
    matching known endpoints have nonpositive undiscounted total credit, even
    across coverage gaps and saturation. This says nothing about discounted PPO
    incentives. Unknown updates and first-ever acquisitions emit no credit.
    """

    def __init__(
        self,
        targets: Sequence[str],
        initial_evidence: Mapping[str, TimingEvidence | None],
        *,
        scaled_min_baseline: float,
        delay_max: float,
    ):
        if not math.isfinite(scaled_min_baseline) or scaled_min_baseline <= 0:
            raise ValueError("scaled_min_baseline must be finite and positive")
        if not math.isfinite(delay_max) or delay_max < 0:
            raise ValueError("delay_max must be finite and nonnegative")
        if not math.isfinite(2.0 * delay_max):
            raise ValueError("timing credit bound 2 * delay_max must be finite")
        self.targets = tuple(targets)
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("timing credit targets must be unique")
        self.delay_max = delay_max
        proxies = {key: forecast_deviation_proxy(initial_evidence.get(key)) for key in targets}
        known = [abs(value) for value in proxies.values() if value is not None]
        count = len(self.targets)
        self.baseline = max(
            math.fsum(value / len(known) for value in known) if known else 0.0,
            scaled_min_baseline / max(1, count),
        )
        if not math.isfinite(self.baseline) or self.baseline <= 0:
            raise ValueError("derived timing baseline must be representable, finite and positive")
        self._anchors = {
            key: self._quality(value) for key, value in proxies.items() if value is not None
        }
        self._previously_known = set(self._anchors)

    def _quality(self, proxy: float) -> float:
        return min(abs(proxy) / self.baseline, 2.0)

    def update(self, evidence_by_target: Mapping[str, TimingEvidence | None]) -> float:
        """Settle consecutive evidence, conservatively settle gaps, retain unknowns."""
        if not self.targets:
            return 0.0
        total = 0.0
        known = set()
        for key in self.targets:
            proxy = forecast_deviation_proxy(evidence_by_target.get(key))
            if proxy is None:
                continue
            known.add(key)
            quality = self._quality(proxy)
            if key in self._anchors:
                delta = self._anchors[key] - quality
                total += delta if key in self._previously_known else min(0.0, delta)
            self._anchors[key] = quality
        self._previously_known = known
        return self.delay_max * (total / len(self.targets))
