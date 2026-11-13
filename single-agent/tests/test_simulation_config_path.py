from pathlib import Path

import pytest
from flight_simulator import MissionDifficulty

from config.config import Config
from config.simulation_config_path import get_config_path


@pytest.mark.parametrize("use_case", [1, 2])
def test_procedure_snapshot_does_not_require_shared_config(tmp_path, monkeypatch, use_case):
    roots = {case: tmp_path / str(case) for case in (1, 2)}
    for root in roots.values():
        (root / "scenario_generator/airports").mkdir(parents=True)
        (root / "designs").mkdir()
    monkeypatch.setattr(Config, "SCENARIO_GENERATOR", "procedure")
    monkeypatch.setattr(Config, "PROCEDURE_CONFIG_ROOTS", roots)
    monkeypatch.setattr(Config, "SIMULATOR_CONFIG_ROOT", str(tmp_path / "absent-shared"))
    difficulty = getattr(MissionDifficulty, f"VALIDATION_USE_CASE_{use_case}")

    path = get_config_path(difficulty=difficulty)

    assert Path(path.simulation_config_root) == roots[use_case]


def test_shared_generator_still_requires_its_config(tmp_path, monkeypatch):
    root = tmp_path / "absent-shared"
    monkeypatch.setattr(Config, "SCENARIO_GENERATOR", "shared")
    monkeypatch.setattr(Config, "SIMULATOR_CONFIG_ROOT", str(root))

    with pytest.raises(FileNotFoundError, match="Scenario config root directory"):
        get_config_path()
