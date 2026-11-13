"""Rebuild the cross-density figures of one checkpoint from stored artifacts.

``attempts-benchmark`` writes one directory per (checkpoint, aircraft count)
cell. This driver picks one checkpoint out of those cells and draws the
comparisons that need several densities on shared axes: pass@k, the
separation-loss risk a retry budget cannot remove, and the nonzero-action
distribution. It re-reads finished runs only - no Ray, no checkpoint, no
sampling - so figures can be restyled or regenerated from
``attempts_summary.json``, ``attempts_raw.csv`` and ``attempts_per_seed.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.runtime_log import force_runtime_log_color, runtime_print
from evaluation.report.density_cells import discover_cells
from evaluation.report.density_plots import (
    render_action_count_density,
    render_exposure_density,
    render_forced_los_density,
    render_passk_density,
)

PASSK_FILENAME = "passk_density.png"
ACTIONS_FILENAME = "action_count_density.png"
FORCED_LOS_FILENAME = "forced_los_density.png"
EXPOSURE_FILENAME = "exposure_density.png"


def main() -> None:
    force_runtime_log_color()
    parser = argparse.ArgumentParser(
        description=(
            "Draw pass@k, separation-loss and nonzero-action figures across traffic densities."
        )
    )
    parser.add_argument(
        "roots",
        type=Path,
        nargs="+",
        metavar="DIR",
        help=(
            "attempts-benchmark output root(s) holding a<aircraft>_<checkpoint> cells; "
            "pass one root per grid, or several roots to combine separate runs."
        ),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        metavar="SLUG",
        help="Checkpoint to plot, matched as a substring of the cell directory name.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory that receives the figures.",
    )
    parser.add_argument(
        "--solve-threshold",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Override the stored success window for the subtitle "
            "(default: use each cell's recorded window)."
        ),
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=4,
        metavar="N",
        help="Histogram bin width in actions (default: 4).",
    )
    parser.add_argument(
        "--exposure-figure",
        action="store_true",
        help=(
            "Also draw the per-density LoS-free@k exposure panel. It answers a different "
            "question - whether any of k attempts infringed - so it rises with k."
        ),
    )
    args = parser.parse_args()

    try:
        cells = discover_cells(args.roots, checkpoint=args.checkpoint)
    except (OSError, ValueError) as exc:
        runtime_print("Density", "error", message=str(exc))
        sys.exit(1)

    label = cells[0].checkpoint
    runtime_print(
        "Density",
        "start",
        checkpoint=label,
        aircraft=[cell.aircraft for cell in cells],
        attempts=[len(cell.attempts) for cell in cells],
    )

    passk_path = args.out_dir / PASSK_FILENAME
    forced_los_path = args.out_dir / FORCED_LOS_FILENAME
    actions_path = args.out_dir / ACTIONS_FILENAME
    render_passk_density(
        passk_path, cells, checkpoint=label, solve_threshold_s=args.solve_threshold
    )
    render_forced_los_density(
        forced_los_path, cells, checkpoint=label, solve_threshold_s=args.solve_threshold
    )
    render_action_count_density(actions_path, cells, checkpoint=label, bin_width=args.bin_width)
    written = [passk_path, forced_los_path, actions_path]
    if args.exposure_figure:
        exposure_path = args.out_dir / EXPOSURE_FILENAME
        render_exposure_density(
            exposure_path, cells, checkpoint=label, solve_threshold_s=args.solve_threshold
        )
        written.append(exposure_path)

    runtime_print("Density", "wrote", paths=[str(path) for path in written])


if __name__ == "__main__":
    main()
