"""Tests for sector polygon rendering helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from atc_env.render_plots import draw_sector_polygons


@pytest.mark.parametrize(
    "colors",
    [
        [(0.1, 0.2, 0.3, 1.0), (0.5, 0.4, 0.2, 1.0)],
        [],
    ],
    ids=["explicit_colors", "fallback_color"],
)
def test_draw_sector_polygons_uses_fill_only_without_borders(colors) -> None:
    fig, ax = plt.subplots()
    try:
        alpha = 0.08
        draw_sector_polygons(
            ax,
            polygons=[
                [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)],
            ],
            colors=colors,
            alpha=alpha,
        )

        assert len(ax.patches) == 2
        bg_rgb = np.asarray(ax.get_facecolor()[:3], dtype=float)
        for patch in ax.patches:
            assert patch.get_linewidth() == 0.0
            assert patch.get_facecolor()[-1] == pytest.approx(1.0)
        if colors:
            for patch, expected in zip(ax.patches, colors, strict=True):
                expected_rgb = (
                    alpha * np.asarray(expected[:3], dtype=float) + (1.0 - alpha) * bg_rgb
                )
                assert patch.get_facecolor()[:3] == pytest.approx(expected_rgb, abs=1e-9)
        else:
            facecolors = [patch.get_facecolor() for patch in ax.patches]
            assert facecolors[0] == facecolors[1]
    finally:
        plt.close(fig)
