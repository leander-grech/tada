"""Pure target-coverage aggregation for numerical timing report dictionaries."""


def summarize_timing_coverage(reports) -> dict:
    """Aggregate target counts; never confuse a non-null episode with full coverage."""
    reports = [report for report in reports if isinstance(report, dict)]
    counts = {
        key: sum(report[key] for report in reports)
        for key in ("target_count", "exact_count", "interval_count", "unknown_count")
    }
    target_count = counts["target_count"]
    return {
        **counts,
        "exact_coverage_fraction": counts["exact_count"] / target_count if target_count else None,
        "coverage_fraction": (counts["exact_count"] + counts["interval_count"]) / target_count
        if target_count
        else None,
        "episodes_with_coverage_report": len(reports),
    }
