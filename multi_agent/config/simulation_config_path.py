"""Module for resolving simulation configuration paths."""

import os

from common.runtime_log import runtime_print
from config.config import Config


# pylint: disable=too-few-public-methods
class SimulationConfigPath:
    """Holds the simulation configuration root path."""

    def __init__(self, simulation_config_root: str):
        self.simulation_config_root = simulation_config_root


def get_config_path():
    rel_root = Config.SIMULATOR_CONFIG_ROOT
    abs_root = os.path.abspath(rel_root)
    runtime_print("ConfigPath", "resolve", relative=rel_root, absolute=abs_root)
    if not os.path.exists(abs_root):
        raise FileNotFoundError(f"Scenario config root directory does not exist: {abs_root}")
    scen_gen_root = os.path.join(abs_root, "scenario_generator")
    runtime_print("ConfigPath", "scenario_generator", path=scen_gen_root)
    if not os.path.exists(scen_gen_root):
        raise FileNotFoundError(
            f"Scenario generator root directory does not exist: {scen_gen_root}"
        )
    airports_root = os.path.join(scen_gen_root, "airports")
    runtime_print("ConfigPath", "airports", path=airports_root)
    if not os.path.exists(airports_root):
        raise FileNotFoundError(f"Airports root directory does not exist: {airports_root}")
    designs_root = os.path.join(abs_root, "designs")
    runtime_print("ConfigPath", "designs", path=designs_root)
    if not os.path.exists(designs_root):
        raise FileNotFoundError(f"Aircraft designs root directory does not exist: {designs_root}")

    return SimulationConfigPath(abs_root)
