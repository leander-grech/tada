"""Authenticate checkpoint-linked development receipts and fixed-denominator coverage."""

import time
from pathlib import Path

from common.atomic_io import atomic_json
from common.hashing import content_digest
from common.jsonio import read_json
from evaluation.benchmark.contract import BenchmarkContract, inference_protocol
from training.checkpoint_manifest import checkpoint_digest


def _identity(checkpoint, contract, manifest):
    from evaluation.checkpoint_adapter import inspect_inference_checkpoint

    run = inspect_inference_checkpoint(checkpoint)
    counters = read_json(checkpoint / "counters.json")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "config_sha256": run.semantic_digest,
        "world_contract_sha256": contract.compatibility_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "world_ids": [w["world_id"] for w in manifest["worlds"]],
        "checkpoint_counters": counters,
        "lifetime_iterations": counters["lifetime_iterations"],
    }


def evaluate_checkpoint(checkpoint: Path, contract: BenchmarkContract, output: Path) -> Path:
    """Pause-side adapter. ``contract.method.development_manifest`` selects inputs.

    The caller owns the supervised deadline and keeps sampling paused throughout.
    Existing receipts are reusable only for this exact complete bundle and inputs.
    """
    from evaluation.benchmark.evaluate import run_development
    from evaluation.benchmark.evidence import (
        AttemptEvidence,
        summarize_world,
        validate_attempt_binding,
    )
    from evaluation.benchmark.worlds import validate_development_manifest
    from training.process import heartbeat

    checkpoint, output = Path(checkpoint), Path(output)
    manifest_path = Path(contract.method["development_manifest"])
    manifest = validate_development_manifest(read_json(manifest_path))
    contract.require_compatible(BenchmarkContract(**manifest["contract"]))
    from evaluation.checkpoint_adapter import inspect_inference_checkpoint

    validate_development_request(inspect_inference_checkpoint(checkpoint), manifest_path)
    identity = _identity(checkpoint, contract, manifest)
    path = output / "evaluation.json"
    if path.exists():
        previous = read_json(path)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("evaluation receipt belongs to a different checkpoint/config/world")
        body = dict(previous)
        if body.pop("evaluation_id", None) != content_digest(body):
            raise ValueError("evaluation receipt content changed")
        return path
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    heartbeat("development-evaluation", checkpoint_sha256=identity["checkpoint_sha256"])
    error = None
    result = None
    try:
        result_path = run_development(checkpoint, manifest_path, output / "attempts")
        result = read_json(result_path)
        body = dict(result)
        if body.pop("result_sha256") != content_digest(body):
            raise ValueError("development result content changed")
        if (
            result["method"]["checkpoint_sha256"] != identity["checkpoint_sha256"]
            or result["method"]["training_config"]["semantic_digest"] != identity["config_sha256"]
        ):
            raise ValueError("development method checkpoint/config mismatch")
        if result["manifest"]["manifest_sha256"] != identity["manifest_sha256"] or result[
            "protocol"
        ] != inference_protocol(k=1, explore=False):
            raise ValueError("development world/protocol mismatch")
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    raw = (
        result["attempts"]
        if result is not None
        else [read_json(p) for p in sorted((output / "attempts" / "attempts").glob("*.json"))]
    )
    worlds = {w["world_id"]: w for w in manifest["worlds"]}
    decoded = []
    seen = set()
    for row in raw:
        attempt = AttemptEvidence.from_dict(row)
        if (
            attempt.world_id not in worlds
            or attempt.world_id in seen
            or attempt.ordinal != 0
            or attempt.mode != "deterministic"
            or attempt.method["checkpoint_sha256"] != identity["checkpoint_sha256"]
        ):
            raise ValueError("unexpected/duplicate development attempt identity")
        validate_attempt_binding(attempt, worlds[attempt.world_id], manifest)
        decoded.append(attempt)
        seen.add(attempt.world_id)
    completed = sum(
        a.completed.diagnostics.get("complete") is True
        and a.completed.diagnostics.get("error") is None
        for a in decoded
    )
    failed = sum(a.completed.diagnostics.get("error") is not None for a in decoded)
    expected = len(worlds)
    coverage = {
        "expected": expected,
        "campaign_required": 100,
        "observed": len(decoded),
        "completed": completed,
        "errors": failed,
        "missing": expected - len(decoded),
        "status": "error"
        if error or failed
        else "full"
        if completed == expected == 100
        else "partial",
    }
    summary = {
        world: summarize_world([a for a in decoded if a.world_id == world], k=1) for world in worlds
    }
    receipt = {
        "version": 1,
        **identity,
        "coverage": coverage,
        "error": error,
        "world_summaries": summary,
        "result_sha256": result["result_sha256"] if result else None,
        "promotion_eligible": coverage["status"] == "full",
        "evaluation_seconds": time.monotonic() - started,
        "claim": "checkpoint-linked development coverage; agent decision required for promotion",
    }
    receipt["evaluation_id"] = content_digest(receipt)
    atomic_json(path, receipt)
    return path


def validate_development_request(run, manifest_path):
    """Refuse unavailable/incompatible development inputs before actor allocation."""
    from evaluation.benchmark.worlds import validate_development_manifest

    manifest = validate_development_manifest(read_json(manifest_path))
    contract = manifest["contract"]["compatibility"]
    from evaluation.benchmark.binding import benchmark_aircraft_count

    if benchmark_aircraft_count(run.env_payload(), contract) != run.scenario["aircraft_count"]:
        raise ValueError("density diagnostics cannot supply primary development receipts")
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    contract = BenchmarkContract(
        manifest["contract"]["compatibility"], {"development_manifest": str(args.manifest)}
    )
    evaluate_checkpoint(args.checkpoint, contract, args.output)


if __name__ == "__main__":
    main()
