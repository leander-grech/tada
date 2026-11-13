"""Record benchmark attempts on the first worlds of a frozen manifest as MP4 videos.

Uses the authenticated inference and attempt sampler of `evaluation.benchmark.evaluate`
(one local runner) and renders every step of each requested attempt;
post-terminal overrun is never recorded. Inference writes a scenario journal next to the
checkpoint, so the checkpoint is copied into the output directory and evaluated from there.
"""

import argparse
import os
import shutil
from pathlib import Path

from common.atomic_io import atomic_json
from common.jsonio import read_json


def video_recorder(output, fps):
    """An RLlib callback class streaming each requested attempt to `seed-<seed>.mp4`."""
    import imageio
    from ray.rllib.callbacks.callbacks import RLlibCallback

    from common.env_utils import get_base_env

    class AttemptVideoRecorder(RLlibCallback):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.writers = {}

        def _requested(self, env):
            env = get_base_env(env)
            return env if env.attempt_id == getattr(env, "requested_attempt_id", None) else None

        def on_episode_start(self, *, env, **kwargs):
            base = self._requested(env)
            # Multi-agent sub-episodes restart within one attempt; keep its writer.
            if base is None or base.attempt_id in self.writers:
                return
            path = Path(output) / f"seed-{base.scenario_seed}.mp4"
            self.writers[base.attempt_id] = imageio.get_writer(
                path, fps=fps, ffmpeg_log_level="quiet", macro_block_size=1
            )
            self.writers[base.attempt_id].append_data(base.render())

        def on_episode_step(self, *, env, **kwargs):
            base = self._requested(env)
            if base is not None and base.attempt_id in self.writers:
                self.writers[base.attempt_id].append_data(base.render())

        def on_episode_end(self, *, env, **kwargs):
            base = self._requested(env)
            if base is None or base.last_end_reason is None or base.attempt_id not in self.writers:
                return
            writer = self.writers.pop(base.attempt_id)
            writer.append_data(base.render())
            writer.close()

    return AttemptVideoRecorder


def record(checkpoint, manifest_path, output, *, worlds, fps, seeds=None):
    """Greedy attempts on the given seeds' worlds, else the first `worlds` manifest worlds;
    returns the video index path."""
    from functools import partial

    import ray

    from evaluation.benchmark.contract import inference_protocol
    from evaluation.benchmark.evaluate import sample_benchmark_shard
    from evaluation.benchmark.worlds import validate_development_manifest
    from evaluation.checkpoint_adapter import build_inference_algorithm
    from training.provenance import checkpoint_method_provenance

    manifest = validate_development_manifest(read_json(manifest_path))
    if seeds:
        selected = [w for w in manifest["worlds"] if w["seed"] in seeds]
        missing = set(seeds) - {w["seed"] for w in selected}
        if missing:
            raise ValueError(f"seeds not in the manifest: {sorted(missing)}")
    else:
        selected = manifest["worlds"][:worlds]
    output = Path(output)
    output.mkdir(parents=True)
    checkpoint = shutil.copytree(checkpoint, output / "checkpoint" / Path(checkpoint).name)
    method = checkpoint_method_provenance(checkpoint)
    ray.init(num_cpus=2, include_dashboard=False)
    algorithm = None
    try:
        algorithm = build_inference_algorithm(
            checkpoint,
            eval_env_config={"evaluation_seeds": [w["seed"] for w in selected]},
            callbacks=[video_recorder(output, fps)],
            timestep_budget=None,
            benchmark_manifest=manifest,
        )
        method["inference"] = algorithm.inference_provenance
        rows = [
            row
            for shard in algorithm.eval_env_runner_group.foreach_env_runner(
                partial(
                    sample_benchmark_shard,
                    manifest=manifest,
                    method=method,
                    protocol=inference_protocol(k=1, explore=False),
                    output=str(output),
                ),
                local_env_runner=True,
            )
            for row in shard
        ]
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
    by_world = {row["world_id"]: row["completed"] for row in rows}
    index = [
        {
            "seed": world["seed"],
            "world_id": world["world_id"],
            "outcome": by_world[world["world_id"]]["outcome"]["kind"],
            "reasons": by_world[world["world_id"]]["outcome"]["reasons"],
            "video": f"seed-{world['seed']}.mp4",
        }
        for world in selected
    ]
    path = output / "videos.json"
    atomic_json(path, {"checkpoint": method["checkpoint_sha256"], "videos": index})
    return path


def main():
    from common.runtime_log import runtime_print

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--worlds", type=int, default=5, help="first manifest worlds to record")
    parser.add_argument("--seeds", type=int, nargs="+", help="record these seeds' worlds instead")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    path = record(
        args.checkpoint,
        args.manifest,
        args.output,
        worlds=args.worlds,
        fps=args.fps,
        seeds=args.seeds,
    )
    runtime_print("RecordVideos", "index", path=str(path))


if __name__ == "__main__":
    main()
