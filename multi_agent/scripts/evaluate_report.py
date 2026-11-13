"""Offline evaluation report: run episodes from a checkpoint and write a PDF."""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from common.runtime_log import force_runtime_log_color, runtime_print
from common.seeding import seed_all
from config.config import Config
from evaluation.loading import build_eval_algo
from evaluation.report.build import build_summary, export_sidecars
from evaluation.report.collect import (
    ReportState,
    build_dataframes,
    make_report_callback,
    reconcile_requested_rows,
)
from evaluation.report.render import write_pdf_report


def _build_output_metadata(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "episodes": int(args.num_episodes),
        "max_aircraft": int(args.max_aircraft),
        "max_steps": int(args.max_steps),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    force_runtime_log_color()
    parser = argparse.ArgumentParser(
        description="Offline evaluation report for a trained PPO checkpoint.",
    )
    parser.add_argument(
        "-d",
        "--model-directory",
        dest="model_dir",
        required=True,
        help="Path to RLlib PPO checkpoint directory.",
    )
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=len(Config.SCENARIO_EVAL_SEEDS),
        help=f"Episodes to run (default: {len(Config.SCENARIO_EVAL_SEEDS)}).",
    )
    parser.add_argument(
        "-a",
        "--max-aircraft",
        type=int,
        default=None,
        help=f"Max aircraft count (default: {Config.MAX_AIRCRAFT_COUNT_EVAL}).",
    )
    parser.add_argument(
        "-m",
        "--max-steps",
        type=int,
        default=None,
        help=f"Max steps per episode (default: {Config.EPISODE_STEPS}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("evaluation_reports"),
        help="Directory for the PDF report and sidecar files.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.model_dir).absolute()
    if not checkpoint.is_dir():
        runtime_print("EvaluateReport", "error", message=f"No such directory: {checkpoint}")
        sys.exit(1)

    from evaluation.checkpoint_adapter import inspect_inference_checkpoint

    saved_run = inspect_inference_checkpoint(checkpoint)
    if args.max_aircraft is None:
        args.max_aircraft = saved_run.scenario["aircraft_count"]
    if args.max_steps is None:
        args.max_steps = saved_run.scenario["episode_steps"]

    import ray

    seed = Config.SEED
    seed_all(seed, deterministic=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray\.")
    ray.init(ignore_reinit_error=True)

    runtime_print(
        "EvaluateReport",
        "start",
        checkpoint=checkpoint,
        episodes=args.num_episodes,
        max_aircraft=args.max_aircraft,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )

    from evaluation.sampling import sample_completed_round
    from evaluation.seeds import operational_seeds

    seeds = operational_seeds(saved_run, args.num_episodes)
    state = ReportState(max_episodes=int(args.num_episodes))
    timestep_budget = int(args.max_steps * args.num_episodes * 1.1)
    algo = build_eval_algo(
        checkpoint,
        eval_env_config={
            "evaluation_mode": True,
            "evaluation_seeds": seeds,
            "max_aircraft_count": args.max_aircraft,
            "max_steps_per_episode": args.max_steps,
            "evaluation_randomize_scenario_count": 0,
        },
        callbacks=[make_report_callback(state)],
        timestep_budget=timestep_budget,
        seed=seed,
    )

    try:
        records = sample_completed_round(algo, seeds=seeds, rng_seed=seed)
    finally:
        algo.stop()
        ray.shutdown()
    reconcile_requested_rows(state, records)

    episode_df, step_df = build_dataframes(state)
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem_path = output_dir / f"evaluation_report_{timestamp}"
    pdf_path = stem_path.with_suffix(".pdf")

    metadata = _build_output_metadata(args, checkpoint)
    summary = build_summary(episode_df, step_df)
    write_pdf_report(
        output_path=pdf_path,
        episode_df=episode_df,
        step_df=step_df,
        metadata=metadata,
    )
    sidecars = export_sidecars(
        stem_path=stem_path,
        episode_df=episode_df,
        step_df=step_df,
        summary={**metadata, **summary},
    )

    runtime_print(
        "EvaluateReport",
        "done",
        pdf=pdf_path,
        episodes_csv=sidecars["episodes"],
        steps_csv=sidecars["steps"],
        summary_json=sidecars["summary"],
    )


if __name__ == "__main__":
    main()
