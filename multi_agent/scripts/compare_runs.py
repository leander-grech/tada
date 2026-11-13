"""Paired world estimands and exact conservative safety uncertainty."""

import math
from pathlib import Path
from typing import Any

import numpy as np

from common.hashing import content_digest
from common.jsonio import read_json
from evaluation.report.density_cells import shared_solve_threshold


def _binomial_cdf(k, n, probability):
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return float(k == n)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(probability)
        + (n - i) * math.log1p(-probability)
        for i in range(k + 1)
    ]
    maximum = max(terms)
    return min(1.0, math.exp(maximum) * math.fsum(math.exp(t - maximum) for t in terms))


def _upper(events, count, alpha):
    if events == count:
        return 1.0
    if events == 0:
        return -math.expm1(math.log(alpha) / count)
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if _binomial_cdf(events, count, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _interval(events, count, alpha):
    return [
        0.0 if events == 0 else 1 - _upper(count - events, count, alpha / 2),
        _upper(events, count, alpha / 2),
    ]


def paired_statistics(rows, weights, *, bootstrap_seed):
    """Bernoulli labels are True/False/None; unknowns use event-count envelopes.

    Resample paired worlds inside each fixed stratum. All retry reduction must
    happen before this boundary: retries are never independent bootstrap units.
    """
    ids = [row["world_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("paired worlds must be nonempty and duplicate-free")
    if (
        set(weights) != {row["stratum"] for row in rows}
        or not all(math.isfinite(w) and w >= 0 for w in weights.values())
        or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("fixed strata must be complete with normalized nonnegative weights")
    if any(
        row[method] is not None and type(row[method]) is not bool
        for row in rows
        for method in ("left", "right")
    ):
        raise ValueError("safety labels must be observed booleans or unknown")
    groups = {
        name: sorted((r for r in rows if r["stratum"] == name), key=lambda r: r["world_id"])
        for name in sorted(weights)
    }
    alpha = 0.05 / (2 * len(groups))
    one_sided = 0.05 / len(groups)
    result: dict[str, Any] = {
        "world_count": len(rows),
        "allocation": {
            "marginal_interval_error": alpha,
            "marginal_tail": alpha / 2,
            "method_upper_tail": one_sided,
        },
        "bootstrap_resamples": 10000,
        "bootstrap_seed": bootstrap_seed,
        "strata": {
            name: {"count": len(group), "weight": weights[name]} for name, group in groups.items()
        },
    }
    for method in ("left", "right"):
        lower = upper = bound = 0.0
        events = unknown = 0
        for name, group in groups.items():
            n = len(group)
            known = sum(r[method] is True for r in group)
            absent = sum(r[method] is None for r in group)
            events += known
            unknown += absent
            lower += weights[name] * _interval(known, n, alpha)[0]
            upper += weights[name] * _interval(known + absent, n, alpha)[1]
            bound += weights[name] * _upper(known + absent, n, one_sided)
        result[method] = {
            "interval": [lower, upper],
            "upper_95": bound,
            "known_events": events,
            "unknown": unknown,
        }
    result["difference_interval"] = [
        result["left"]["interval"][0] - result["right"]["interval"][1],
        result["left"]["interval"][1] - result["right"]["interval"][0],
    ]
    result.update(point_difference=None, bootstrap_interval=None, bootstrap_degenerate=None)
    if result["left"]["unknown"] or result["right"]["unknown"]:
        return result
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.zeros(10000)
    point = 0.0
    for name, group in groups.items():
        differences = np.array([int(r["left"]) - int(r["right"]) for r in group])
        point += weights[name] * float(differences.mean())
        boot += weights[name] * differences[
            rng.integers(0, len(group), size=(10000, len(group)))
        ].mean(axis=1)
    interval = np.percentile(boot, [2.5, 97.5]).tolist()
    result.update(
        point_difference=point,
        bootstrap_interval=interval,
        bootstrap_degenerate=interval[0] == interval[1],
    )
    return result


def compare_results(left, right, *, bootstrap_seed: int, solve_threshold_s: float) -> dict:
    """Compare canonical exports only; unmatched/scalar historical runs fail closed.

    Both exports must have been graded with the requested ``solve_threshold_s``:
    outcomes graded with different windows are not comparable.
    """
    from evaluation.benchmark.contract import (
        BenchmarkContract,
        validate_inference_protocol,
        validate_manifest,
    )
    from evaluation.benchmark.evidence import (
        AttemptEvidence,
        summarize_world,
        validate_attempt_binding,
    )

    documents = []
    summaries = []
    for path in (left, right):
        document = read_json(Path(path))
        body = dict(document)
        digest = body.pop("result_sha256", None)
        if digest != content_digest(body) or "attempts" not in body:
            raise ValueError(
                "canonical attempt export required; historical scalar result is unscorable"
            )
        manifest = validate_manifest(document["manifest"])
        validate_inference_protocol(document["protocol"])
        declared = {w["world_id"] for w in manifest["worlds"]}
        attempts = [AttemptEvidence.from_dict(a) for a in document["attempts"]]
        worlds_by_id = {w["world_id"]: w for w in manifest["worlds"]}
        for attempt in attempts:
            if attempt.world_id not in worlds_by_id:
                raise ValueError("attempt references an undeclared world")
            validate_attempt_binding(attempt, worlds_by_id[attempt.world_id], manifest)
        if any(
            a.world_id not in declared
            or a.benchmark_id != manifest["compatibility_sha256"]
            or content_digest(a.method) != content_digest(document["method"])
            or a.mode != document["protocol"]["mode"]
            for a in attempts
        ):
            raise ValueError("attempt does not belong to the declared world/method/protocol")
        summaries.append(
            {
                identity: summarize_world(
                    [a for a in attempts if a.world_id == identity], k=document["protocol"]["k"]
                )
                for identity in declared
            }
        )
        documents.append(document)
    a, b = documents
    graded = shared_solve_threshold(
        (d["manifest"]["contract"]["compatibility"]["threshold_s"] for d in documents),
        "compared exports",
    )
    if graded != solve_threshold_s:
        raise ValueError(
            f"exports were graded at {graded:g} s, not the requested {solve_threshold_s:g} s"
        )
    BenchmarkContract(**a["manifest"]["contract"]).require_compatible(
        BenchmarkContract(**b["manifest"]["contract"])
    )
    if (
        a["manifest"]["worlds"] != b["manifest"]["worlds"]
        or a["protocol"] != b["protocol"]
        or a["budget_label"] != b["budget_label"]
    ):
        raise ValueError("missing/incompatible worlds, inference protocol or compute budget")
    contract = a["manifest"]["contract"]["compatibility"]
    strata = contract["strata"]
    if len(strata) != 1:
        raise ValueError("multi-stratum export must explicitly bind every world stratum")
    weights = {s["id"]: s["weight"] for s in strata}
    statistics = {}
    for metric in (
        "any_eligible_success",
        "any_attempt_hazard",
        "any_verifier_hazard",
        "selected_candidate_hazard",
    ):
        rows = [
            {
                "world_id": w["world_id"],
                "stratum": strata[0]["id"],
                "left": summaries[0][w["world_id"]][metric],
                "right": summaries[1][w["world_id"]][metric],
            }
            for w in a["manifest"]["worlds"]
        ]
        statistics[metric] = paired_statistics(rows, weights, bootstrap_seed=bootstrap_seed)
    return {
        "matched": True,
        "solve_threshold_s": graded,
        "estimate_label": a["estimate_label"],
        "protocol": a["protocol"],
        "budget_label": a["budget_label"],
        "parent_compute": {"left": a["parent_compute"], "right": b["parent_compute"]},
        "statistics": statistics,
        "timing": "retained per-world intervals/nulls; no invented exact means",
        "assumptions": "fixed-method Bernoulli worlds within fixed strata; development is adaptive diagnostic only",
    }


def main() -> None:
    import argparse

    from common.atomic_io import atomic_json

    parser = argparse.ArgumentParser(description="Paired comparison of two canonical exports.")
    parser.add_argument("baseline", help="canonical results.json of the baseline run")
    parser.add_argument("candidate", help="canonical results.json of the candidate run")
    parser.add_argument(
        "--solve-threshold",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Grading window both exports must share; mixed windows are refused.",
    )
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--output", required=True, help="new comparison JSON path")
    args = parser.parse_args()
    comparison = compare_results(
        args.baseline,
        args.candidate,
        bootstrap_seed=args.bootstrap_seed,
        solve_threshold_s=args.solve_threshold,
    )
    atomic_json(Path(args.output), comparison, exclusive=True)


if __name__ == "__main__":
    main()
