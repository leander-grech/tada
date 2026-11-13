import os
from config.config import Config
from flight_simulator import MissionDifficulty

class SimulationConfigPath:
    def __init__(self, simulation_config_root: str):
        self.simulation_config_root = simulation_config_root

def get_config_path(verbose=False, *, difficulty=None):
    rel_root = Config.SIMULATOR_CONFIG_ROOT
    if Config.SCENARIO_GENERATOR == "procedure":
        difficulty = Config.DEFAULT_EVALUATION_DIFFICULTY if difficulty is None else difficulty
        use_case = {
            MissionDifficulty.VALIDATION_USE_CASE_1: 1,
            MissionDifficulty.VALIDATION_USE_CASE_2: 2,
        }.get(difficulty)
        if use_case is None:
            raise ValueError(f"Procedure generator has no snapshot for difficulty {difficulty}")
        rel_root = Config.PROCEDURE_CONFIG_ROOTS[use_case]
    abs_root = os.path.abspath(rel_root)
    if verbose: print(f"Resolving config root, relative path from config.py: {rel_root}, absolute path: {abs_root}")
    if not os.path.exists(abs_root):
        raise FileNotFoundError(f"Scenario config root directory does not exist: {abs_root}")
    scen_gen_root = os.path.join(abs_root, "scenario_generator")
    if verbose: print(f"Scenario generator root: {scen_gen_root}")
    if not os.path.exists(scen_gen_root):
        raise FileNotFoundError(f"Scenario generator root directory does not exist: {scen_gen_root}")
    airports_root = os.path.join(scen_gen_root, "airports")
    if verbose: print(f"Airports root: {airports_root}")
    if not os.path.exists(airports_root):
        raise FileNotFoundError(f"Airports root directory does not exist: {airports_root}")
    designs_root = os.path.join(abs_root, "designs")
    if verbose: print(f"Aircraft designs root: {designs_root}")
    if not os.path.exists(designs_root):
        raise FileNotFoundError(f"Aircraft designs root directory does not exist: {designs_root}")

    return SimulationConfigPath(abs_root)
