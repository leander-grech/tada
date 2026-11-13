"""Pure deterministic top-K risk ordering; caller supplies physical thresholds."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NeighborRisk:
    callsign: str
    forecast_severity: float
    severe_forecast: bool
    current_separation_threat: bool
    raw_cpa_s: float
    cpa_horizontal_distance_nm: float
    vertical_relevance: float  # abs(vertical delta ft) / vertical safezone ft
    range_nm: float


def neighbor_priority(risk: NeighborRisk) -> tuple:
    """Prefer severe, nearby protective-band, future-CPA, then remaining traffic.

    Vertical-band relevance precedes horizontal CPA distance. Only positive
    finite CPA is a future minimum; opening/stationary traffic uses its current
    range. Raw seconds, never encoded/clamped times, resolve physical ties.
    """
    approaching = math.isfinite(risk.raw_cpa_s) and risk.raw_cpa_s > 0
    category = (
        0
        if risk.severe_forecast
        else 1
        if risk.current_separation_threat
        else 2
        if approaching
        else 3
    )
    future_distance = risk.cpa_horizontal_distance_nm if approaching else risk.range_nm
    future_time = risk.raw_cpa_s if approaching else math.inf
    return (
        category,
        -risk.forecast_severity if risk.severe_forecast else 0.0,
        risk.vertical_relevance > 1.0,
        future_distance,
        risk.vertical_relevance,
        future_time,
        risk.range_nm,
        risk.callsign,
    )
