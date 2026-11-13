"""World-clustered uncertainty, including unknown safety labels."""

import math

import pytest

from scripts.compare_runs import paired_statistics


def sample(left, right, strata=None):
    return [
        {"world_id": str(i), "stratum": strata[i] if strata else "common", "left": a, "right": b}
        for i, (a, b) in enumerate(zip(left, right, strict=True))
    ]


@pytest.mark.parametrize("n", [1, 7, 17, 100])
def test_no_events_has_positive_exact_uncertainty(n):
    result = paired_statistics(sample([False] * n, [False] * n), {"common": 1.0}, bootstrap_seed=19)
    upper = 1 - 0.0125 ** (1 / n)
    assert result["bootstrap_degenerate"]
    assert result["difference_interval"] == pytest.approx([-upper, upper])
    assert result["left"]["upper_95"] == pytest.approx(1 - 0.05 ** (1 / n))
    assert result["left"]["upper_95"] > 0


def test_all_events_and_one_discordance_are_finite_and_replayable():
    for rows in (sample([True], [True]), sample([True, False, True], [False, False, True])):
        result = paired_statistics(rows, {"common": 1.0}, bootstrap_seed=71)
        assert result == paired_statistics(list(reversed(rows)), {"common": 1.0}, bootstrap_seed=71)
        assert all(math.isfinite(v) for v in result["difference_interval"])
        assert result["difference_interval"][0] < result["difference_interval"][1]


def test_strata_use_frozen_weights_and_correct_error_allocation():
    result = paired_statistics(
        sample([False] * 8, [True] * 8, ["a"] * 7 + ["b"]), {"a": 0.2, "b": 0.8}, bootstrap_seed=3
    )
    expected_upper = 0.2 * (1 - 0.00625 ** (1 / 7)) + 0.8 * (1 - 0.00625)
    assert result["left"]["interval"][1] == pytest.approx(expected_upper)
    assert result["point_difference"] == -1
    assert result["left"]["upper_95"] == pytest.approx(0.2 * (1 - 0.025 ** (1 / 7)) + 0.8 * 0.975)


def test_unknown_is_not_zero_and_denominator_is_unchanged():
    result = paired_statistics(sample([None] * 7, [False] * 7), {"common": 1.0}, bootstrap_seed=9)
    assert result["left"]["interval"] == [0, 1]
    assert result["left"]["upper_95"] == 1
    assert result["left"]["unknown"] == 7
    assert result["world_count"] == 7
    assert result["point_difference"] is None
    assert result["bootstrap_interval"] is None


def test_duplicate_worlds_missing_strata_and_bad_weights_fail():
    rows = sample([False, True], [True, False])
    for invalid_rows, weights in (
        (rows + [rows[0]], {"common": 1.0}),
        (rows, {"common": 0.4}),
        (rows, {"common": 0.5, "missing": 0.5}),
    ):
        with pytest.raises(ValueError):
            paired_statistics(invalid_rows, weights, bootstrap_seed=1)


def test_canonical_comparison_accepts_method_difference_rejects_world_drift(tmp_path):
    import json

    from common.hashing import content_digest
    from evaluation.benchmark.contract import BenchmarkContract, inference_protocol
    from scripts.compare_runs import compare_results

    contract = BenchmarkContract(
        {
            "threshold_s": 60,
            "verification_cap_s": 60,
            "strata": [{"id": "common", "weight": 1.0, "count": 2}],
        },
        {},
    )
    worlds: list[dict] = [{"seed": seed, "raw_world": {"aircraft": [seed]}} for seed in (7, 11)]
    for world in worlds:
        world["world_id"] = content_digest(world)
    manifest = {
        "contract": contract.to_dict(),
        "compatibility_sha256": contract.compatibility_sha256,
        "worlds": worlds,
    }
    manifest["manifest_sha256"] = content_digest(manifest)

    def export(name, manifest):
        body = {
            "manifest": manifest,
            "method": {"checkpoint_sha256": name},
            "protocol": inference_protocol(k=1, explore=False),
            "attempts": [],
            "budget_label": "matched-attempt-budget",
            "estimate_label": "adaptive-development-diagnostic",
            "parent_compute": {"wall_seconds": 1},
        }
        body["result_sha256"] = content_digest(body)
        path = tmp_path / name
        path.write_text(json.dumps(body))
        return path

    left = export("a", manifest)
    right = export("b", manifest)
    result = compare_results(left, right, bootstrap_seed=4, solve_threshold_s=60)
    assert result["matched"]
    assert result["solve_threshold_s"] == 60
    with pytest.raises(ValueError, match="graded at 60 s, not the requested 30 s"):
        compare_results(left, right, bootstrap_seed=4, solve_threshold_s=30)
    assert result["statistics"]["any_attempt_hazard"]["left"]["upper_95"] == 1
    changed = json.loads(json.dumps(manifest))
    changed["worlds"].pop()
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = content_digest(changed)
    with pytest.raises(ValueError, match="world"):
        compare_results(left, export("c", changed), bootstrap_seed=4, solve_threshold_s=60)
    scalar = tmp_path / "scalar"
    scalar.write_text(json.dumps({"mean_score": 1}))
    with pytest.raises(ValueError, match="scalar"):
        compare_results(left, scalar, bootstrap_seed=4, solve_threshold_s=60)


def test_fixed_stratum_zero_weight_does_not_change_estimand():
    result = paired_statistics(
        sample([False, True], [True, False], ["a", "b"]), {"a": 0.0, "b": 1.0}, bootstrap_seed=2
    )
    assert result["point_difference"] == 1.0


def test_mixed_grading_windows_are_refused():
    from evaluation.report.density_cells import shared_solve_threshold

    assert shared_solve_threshold([60, 60.0], "exports") == 60
    with pytest.raises(ValueError, match="exports must share a solve threshold"):
        shared_solve_threshold([60, 30], "exports")
