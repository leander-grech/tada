"""Pure coverage views; proxy magnitudes never become numerical arrival reports."""

from outcomes.evidence import TimingEvidence, forecast_deviation_proxy
from outcomes.ledger import OutcomeSnapshot


def timing_map(snapshot: OutcomeSnapshot | None) -> dict[str, TimingEvidence | None]:
    return {} if snapshot is None else {a.callsign: a.timing for a in snapshot.arrivals}


def proxy_map(snapshot: OutcomeSnapshot | None) -> dict[str, float]:
    return {
        callsign: proxy
        for callsign, evidence in timing_map(snapshot).items()
        if (proxy := forecast_deviation_proxy(evidence)) is not None
    }


def exact_map(snapshot: OutcomeSnapshot | None) -> dict[str, float]:
    return {
        callsign: value.lower_error_s
        for callsign, value in timing_map(snapshot).items()
        if value is not None
        and value.lower_error_s == value.upper_error_s
        and forecast_deviation_proxy(value) is not None
    }


def success_margin(proxies: dict[str, float], targets: set[str], threshold: float) -> float:
    if not targets or not targets <= proxies.keys():
        return 0.0
    return sum(max(0.0, min(1.0, 1.0 - abs(proxies[t]) / threshold)) for t in targets) / len(
        targets
    )


def format_timing(value: TimingEvidence | None) -> str:
    if forecast_deviation_proxy(value) is None:
        return "unknown"
    assert value is not None
    if value.lower_error_s == value.upper_error_s:
        return f"{value.lower_error_s:+.0f} s"
    return f"[{value.lower_error_s:+.0f}, {value.upper_error_s:+.0f}] s"


def timing_report(snapshot: OutcomeSnapshot | None) -> dict:
    """Numerical reporting never reports a proxy endpoint as a measured arrival."""
    evidence = timing_map(snapshot)
    exact = exact_map(snapshot)
    intervals = {
        key: value
        for key, value in evidence.items()
        if value is not None and key not in exact and forecast_deviation_proxy(value) is not None
    }
    known = len(exact) + len(intervals)
    return {
        "exact_total_absolute_error_s": sum(abs(value) for value in exact.values())
        if evidence and len(exact) == len(evidence)
        else None,
        "target_count": len(evidence),
        "exact_coverage_fraction": len(exact) / len(evidence) if evidence else 0.0,
        "exact_count": len(exact),
        "interval_count": len(intervals),
        "unknown_count": len(evidence) - known,
        "sign_ambiguous_count": sum(
            value.lower_error_s < 0 < value.upper_error_s for value in intervals.values()
        ),
        "coverage_fraction": known / len(evidence) if evidence else 0.0,
        "exact_mean_absolute_error_s": sum(abs(value) for value in exact.values()) / len(exact)
        if exact
        else None,
        "arrivals": {
            key: (
                {"kind": "exact", "error_s": exact[key]}
                if key in exact
                else {
                    "kind": "interval",
                    "lower_error_s": intervals[key].lower_error_s,
                    "upper_error_s": intervals[key].upper_error_s,
                }
                if key in intervals
                else {"kind": "unknown"}
            )
            for key in evidence
        },
    }
