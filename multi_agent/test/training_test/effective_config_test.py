"""Execution controls must not alter experiment semantic compatibility."""

import json

import pytest

from training.run_config import load_run_config


def test_execution_is_immutable_operational_and_recorded(config_path):
    first = load_run_config(config_path)
    data = json.loads(config_path.read_text())
    data["execution"] = {"stop_iterations": 2, "output_root": "diagnostic-output"}
    config_path.write_text(json.dumps(data))
    second = load_run_config(config_path)
    assert first.semantic_digest == second.semantic_digest
    assert first.env_payload() == second.env_payload()
    assert second.execution["output_root"] == str(config_path.parent / "diagnostic-output")
    assert second.to_dict()["execution"]["stop_iterations"] == 2
    with pytest.raises(TypeError):
        second.execution["stop_iterations"] = 3


@pytest.mark.parametrize(
    "execution",
    [
        {"stop_iterations": 0},
        {"stop_iterations": True},
        {"stop_iterations": 1.5},
        {"output_root": ""},
        {"max_env_steps": 10},
    ],
)
def test_execution_rejects_unsupported_controls(config_path, execution):
    data = json.loads(config_path.read_text())
    data["execution"] = execution
    config_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="execution|stop_iterations"):
        load_run_config(config_path)
