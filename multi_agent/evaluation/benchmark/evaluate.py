"""Checkpoint-linked development evaluation; no heldout opening entry point."""

import time
from pathlib import Path

from common.atomic_io import atomic_json
from common.device import DEVICES
from common.hashing import content_digest
from common.jsonio import read_json
from evaluation.benchmark.contract import attempt_key, attempt_rng_ids, inference_protocol
from evaluation.benchmark.evidence import AttemptEvidence, summarize_world, validate_attempt_binding


def sample_benchmark_shard(runner, *, manifest, method, protocol, output):
    from common.env_utils import get_base_env
    from common.runtime_log import runtime_print
    from evaluation.sampling import sample_attempt

    env = get_base_env(runner.env)
    runner.module["shared_policy"].eval()
    benchmark_id = manifest["compatibility_sha256"]
    records = []
    for world in manifest["worlds"]:
        if world["seed"] not in env._eval_seeds:
            continue
        for ordinal in range(protocol["k"]):
            rng = attempt_rng_ids(benchmark_id, world["world_id"], ordinal)
            identity = attempt_key(
                benchmark_id,
                world["world_id"],
                protocol["mode"],
                ordinal,
                method["checkpoint_sha256"],
            )
            runner._validate_boundary_layout()
            env.benchmark_request = {"rng": rng}
            result = sample_attempt(
                runner,
                env,
                seed=world["seed"],
                attempt_id=identity,
                explore=protocol["mode"] == "stochastic",
            )
            # Nothing reads benchmark runner metrics; draining releases each attempt's
            # retained RLlib episodes, which otherwise grow until Ray's OOM killer acts.
            runner.get_metrics()
            evidence = AttemptEvidence(
                benchmark_id,
                world["world_id"],
                ordinal,
                protocol["mode"],
                rng,
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
            row = evidence.to_dict()
            validate_attempt_binding(evidence, world, manifest)
            atomic_json(Path(output) / "attempts" / f"{identity}.json", row, exclusive=True)
            records.append(row)
            runtime_print(
                "Benchmark",
                "development attempt",
                seed=world["seed"],
                ordinal=ordinal,
                complete=result.diagnostics.get("complete"),
                eligible=result.eligible,
                error=result.diagnostics.get("error"),
            )
    return records


def run_development(
    checkpoint,
    manifest_path,
    output,
    *,
    k=1,
    explore=False,
    workers=0,
    inference_device="cpu",
    gpus_per_runner=0.0,
    policy_time_limit_scale=None,
):
    from functools import partial

    import ray

    from evaluation.inference_resources import InferenceResources

    resources = InferenceResources(inference_device, workers, gpus_per_runner)
    resources.preflight()
    from evaluation.benchmark.worlds import validate_development_manifest
    from evaluation.checkpoint_adapter import build_inference_algorithm
    from training.provenance import checkpoint_method_provenance

    protocol = inference_protocol(k=k, explore=explore)
    manifest = validate_development_manifest(read_json(manifest_path))
    method = checkpoint_method_provenance(checkpoint)
    env_options = {}
    if policy_time_limit_scale is not None:
        # Protocol variant, not the saved benchmark: recorded in the method provenance.
        from evaluation.checkpoint_adapter import inspect_inference_checkpoint

        saved = inspect_inference_checkpoint(checkpoint).scenario["episode_steps"]
        steps = round(policy_time_limit_scale * saved)
        method["policy_time_limit_steps"] = env_options["policy_time_limit_steps"] = steps
    owned_ray = not ray.is_initialized()
    if inference_device == "cuda" and not owned_ray:
        raise ValueError("CUDA benchmark requires a fresh owned Ray cluster")
    output = Path(output)
    if output.exists():
        raise ValueError("development result directory already exists; use a new explicit run")
    output.mkdir(parents=True)
    algorithm = None
    started = time.monotonic()
    try:
        if owned_ray:
            ray.init(
                num_cpus=max(2, workers + 1), num_gpus=resources.ray_gpus, include_dashboard=False
            )
        if resources.ray_gpus > ray.available_resources().get("GPU", 0):
            raise ValueError("insufficient available Ray GPU capacity for inference")
        algorithm = build_inference_algorithm(
            checkpoint,
            eval_env_config={
                "evaluation_seeds": [w["seed"] for w in manifest["worlds"]],
                "evaluation_seed_shards": workers,
                **env_options,
            },
            callbacks=[],
            timestep_budget=None,
            explore=explore,
            num_env_runners=workers,
            benchmark_manifest=manifest,
            inference_device=inference_device,
            gpus_per_runner=gpus_per_runner,
        )
        method["inference"] = algorithm.inference_provenance
        group = algorithm.eval_env_runner_group
        records = [
            row
            for shard in group.foreach_env_runner(
                partial(
                    sample_benchmark_shard,
                    manifest=manifest,
                    method=method,
                    protocol=protocol,
                    output=str(output),
                ),
                local_env_runner=group.num_healthy_remote_workers() == 0,
            )
            for row in shard
        ]
        decoded = [AttemptEvidence.from_dict(row) for row in records]
        expected = {(w["world_id"], i) for w in manifest["worlds"] for i in range(k)}
        actual = [(a.world_id, a.ordinal) for a in decoded]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("missing/duplicate immutable benchmark transport")
        summaries = {
            w["world_id"]: summarize_world([a for a in decoded if a.world_id == w["world_id"]], k=k)
            for w in manifest["worlds"]
        }
        result = {
            "version": 1,
            "manifest": manifest,
            "method": method,
            "protocol": protocol,
            "attempts": records,
            "world_summaries": summaries,
            "estimate_label": "adaptive-development-diagnostic",
            "budget_label": "matched-attempt-budget",
            "parent_compute": {
                "wall_seconds": time.monotonic() - started,
                "workers": workers,
                "k": k,
            },
            "training_resume": False,
        }
        result["result_sha256"] = content_digest(result)
        atomic_json(output / "results.json", result, exclusive=True)
        return output / "results.json"
    finally:
        try:
            if algorithm is not None:
                algorithm.stop()
        finally:
            if owned_ray:
                ray.shutdown()


def main():
    import argparse

    from common.runtime_log import runtime_print

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--inference-device", choices=DEVICES, default="cpu")
    parser.add_argument("--gpus-per-runner", type=float, default=0.0)
    parser.add_argument(
        "--policy-time-limit-scale",
        type=float,
        default=1.5,
        help="evaluation-only time limit: let the policy act for this multiple of the saved "
        "episode length so attempts resolve instead of timing out (1 = saved length)",
    )
    args = parser.parse_args()
    if args.inference_device == "cpu":
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    path = run_development(
        args.checkpoint,
        args.manifest,
        args.output,
        k=args.k,
        explore=args.stochastic,
        workers=args.workers,
        inference_device=args.inference_device,
        gpus_per_runner=args.gpus_per_runner,
        policy_time_limit_scale=args.policy_time_limit_scale,
    )
    runtime_print("Benchmark", "results", path=str(path))


if __name__ == "__main__":
    main()
