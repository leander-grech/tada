"""Tests for selected-sector render coloring behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from support.env_patch import patch_env_global
from support.settings import live_env_settings

from atc_env.environment import ATCEnv


def test_resolve_render_sector_polygons_unifies_color_within_selected_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()

    patch_env_global(monkeypatch, "get_action_selection_mode", lambda _settings: "sector_based")
    patch_env_global(
        monkeypatch,
        "get_sector_selection_groups",
        lambda _settings: (("ALPHA", "BRAVO"), ("CHARLIE",)),
    )
    patch_env_global(
        monkeypatch,
        "get_selected_group_index_cached_from_normalized",
        lambda _groups: SimpleNamespace(
            selected_groups=(("ALPHA", "BRAVO"), ("CHARLIE",)),
            sector_order=("ALPHA", "BRAVO", "CHARLIE"),
            sector_to_group={
                "ALPHA": ("ALPHA", "BRAVO"),
                "BRAVO": ("ALPHA", "BRAVO"),
                "CHARLIE": ("CHARLIE",),
            },
            sector_defs={
                "ALPHA": {"polygon": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]},
                "BRAVO": {"polygon": [(1.0, 0.0), (2.0, 0.0), (1.0, 1.0)]},
                "CHARLIE": {"polygon": [(2.0, 0.0), (3.0, 0.0), (2.0, 1.0)]},
            },
        ),
    )

    _polygons, colors = ATCEnv._resolve_render_sector_polygons(env)

    assert colors[0] == colors[1]
    assert colors[0] != colors[2]


def test_resolve_render_sector_polygons_draws_in_group_precedence_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = ATCEnv.__new__(ATCEnv)
    env.settings = live_env_settings()

    patch_env_global(monkeypatch, "get_action_selection_mode", lambda _settings: "sector_based")
    patch_env_global(
        monkeypatch,
        "get_sector_selection_groups",
        lambda _settings: (("LOW",), ("HIGH",)),
    )
    patch_env_global(
        monkeypatch,
        "get_selected_group_index_cached_from_normalized",
        lambda _groups: SimpleNamespace(
            selected_groups=(("LOW",), ("HIGH",)),
            # Selection checks sectors in reverse precedence order (HIGH first).
            sector_order=("HIGH", "LOW"),
            sector_to_group={
                "LOW": ("LOW",),
                "HIGH": ("HIGH",),
            },
            sector_defs={
                # Intentional overlap to validate draw precedence.
                "LOW": {"polygon": [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]},
                "HIGH": {"polygon": [(1.0, -0.5), (3.0, -0.5), (1.0, 1.5)]},
            },
        ),
    )

    polygons, colors = ATCEnv._resolve_render_sector_polygons(env)

    # LOW group is rendered first, HIGH group last (on top).
    assert polygons[0][0] == (0.0, 0.0)
    assert polygons[1][0] == (1.0, -0.5)
    assert colors[0] != colors[1]
