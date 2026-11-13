"""Canonical export readers authenticate physical attempt, RNG and protocol keys."""

import copy
import json
from typing import Any

import pytest

from atc_env.completed_scenario import CompletedScenarioResult
from common.hashing import content_digest
from evaluation.benchmark.contract import BenchmarkContract, attempt_rng_id
from evaluation.benchmark.evaluate import inference_protocol
from evaluation.benchmark.evidence import AttemptEvidence, summarize_world
from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
from outcomes.ledger import OutcomeSnapshot
from outcomes.verification import classify_outcome
from scripts.compare_runs import compare_results


def canonical_export(checkpoint):
    contract = BenchmarkContract(
        {
            "threshold_s": 60,
            "verification_cap_s": 6000,
            "strata": [{"id": "common", "count": 1, "weight": 1.0}],
        },
        {},
    )
    world: dict[str, Any] = {
        "seed": 7,
        "raw_world": {"fixture": "raw"},
        "policy_start_world": {"fixture": "policy"},
        "effective_controller": {},
        "targets": [{"callsign": "AC1", "eta_s": 123.25}],
    }
    world["world_id"] = content_digest(world)
    manifest = {
        "contract": contract.to_dict(),
        "compatibility_sha256": contract.compatibility_sha256,
        "worlds": [world],
    }
    manifest["manifest_sha256"] = content_digest(manifest)
    method = {
        "checkpoint_sha256": checkpoint,
        "training_config": {
            "reserved_manifest": {"validation": [{"seed": 7}]},
            "features": {"dimensions": ["eta", "altitude"]},
        },
    }
    ledger = OutcomeSnapshot(
        (ArrivalTargetSpec("AC1", 123.25),),
        (ArrivalEvidence("AC1", True, True, TimingEvidence(0, 0, "fixture")),),
        (),
        (),
        False,
    )
    attempts = []
    # Ordinal 1 is genuinely missing; ordinal 2 is an observed failed attempt.
    for ordinal in (0, 2):
        identity = content_digest(
            [contract.compatibility_sha256, world["world_id"], "stochastic", ordinal, checkpoint]
        )
        result = CompletedScenarioResult(
            identity,
            classify_outcome(ledger, None, threshold_s=60, policy_limit_reached=True),
            ledger,
            None,
            [{"action": 8, "phase": "pre_episode"}, {"action": 1}],
            0,
            {
                "complete": ordinal == 0,
                "error": None if ordinal == 0 else "fixture error",
                "steps": 2,
                "grading_threshold_s": 60,
                "input_binding": {
                    "world_id": world["world_id"],
                    "manifest_sha256": manifest["manifest_sha256"],
                },
            },
        )
        attempt = AttemptEvidence(
            contract.compatibility_sha256,
            world["world_id"],
            ordinal,
            "stochastic",
            {
                domain: attempt_rng_id(
                    contract.compatibility_sha256, world["world_id"], ordinal, domain
                )
                for domain in ("policy", "frame")
            },
            method,
            {
                "world_id": world["world_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "raw_world_sha256": content_digest(world["raw_world"]),
                "policy_start_sha256": content_digest(world["policy_start_world"]),
                "effective_controller_sha256": content_digest(world["effective_controller"]),
            },
            result,
        )
        attempts.append(attempt.to_dict())
    return {
        "manifest": manifest,
        "method": method,
        "protocol": inference_protocol(k=3, explore=True),
        "attempts": attempts,
        "budget_label": "matched-attempt-budget",
        "estimate_label": "adaptive-development-diagnostic",
        "parent_compute": {"wall_seconds": 1},
    }


def write_export(path, document):
    for row in document["attempts"]:
        row.pop("evidence_sha256", None)
        row["evidence_sha256"] = content_digest(row)
    document.pop("result_sha256", None)
    document["result_sha256"] = content_digest(document)
    path.write_text(json.dumps(document))
    return path


def test_nonempty_distinct_methods_missing_and_error_round_trip(tmp_path):
    left, right = canonical_export("a"), canonical_export("b")
    summary = summarize_world([AttemptEvidence.from_dict(a) for a in left["attempts"]], k=3)
    assert summary["observed_attempt_count"] == 2
    assert summary["missing_count"] == 1
    assert summary["command_cost_all_attempts"] == 4
    assert summary["policy_command_count_all_attempts"] == 2
    assert summary["autosolve_command_count_all_attempts"] == 2
    assert summary["policy_steps_all_attempts"] == 4
    assert summary["any_attempt_hazard"] is None
    result = compare_results(
        write_export(tmp_path / "a", left),
        write_export(tmp_path / "b", right),
        bootstrap_seed=19,
        solve_threshold_s=60,
    )
    assert result["matched"]
    assert result["statistics"]["any_attempt_hazard"]["left"]["unknown"] == 1
    assert result["statistics"]["any_attempt_hazard"]["left"]["upper_95"] == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "relabeled_duplicate",
        "wrong_rng",
        "absent_rng",
        "wrong_key",
        "deterministic_k",
        "nested_method",
        "early_stop",
    ],
)
def test_rehashed_canonical_export_rejects_semantic_identity_corruption(tmp_path, corruption):
    document = canonical_export("a")
    first = document["attempts"][0]
    if corruption == "relabeled_duplicate":
        duplicate = copy.deepcopy(first)
        duplicate["ordinal"] = 1
        document["attempts"].append(duplicate)
    elif corruption == "wrong_rng":
        first["rng"]["policy"] += 1
    elif corruption == "absent_rng":
        first["rng"] = {}
    elif corruption == "wrong_key":
        first["completed"]["attempt_id"] = "unrelated-physical-record"
    elif corruption == "deterministic_k":
        document["attempts"] = [first]
        first["mode"] = "deterministic"
        first["completed"]["attempt_id"] = content_digest(
            [first["benchmark_id"], first["world_id"], "deterministic", 0, "a"]
        )
        document["protocol"] = {**inference_protocol(k=1, explore=False), "k": 2}
    elif corruption == "nested_method":
        document["method"]["training_config"]["features"]["dimensions"][0] = "changed"
    else:
        document["protocol"]["early_stop"] = True
    path = write_export(tmp_path / "corrupted", document)
    with pytest.raises(ValueError):
        compare_results(path, path, bootstrap_seed=19, solve_threshold_s=60)
