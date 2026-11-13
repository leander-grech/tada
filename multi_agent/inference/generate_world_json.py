"""Generate a simulator world JSON for inference testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.runtime_log import runtime_print
from config.config import Config
from config.settings import settings_from_config
from config.simulation_config_path import get_config_path
from simulator.route_section_store import RouteSectionStore
from simulator.simulator import Simulator

_COMPONENT = "GenerateWorld"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4200, help="Scenario seed.")
    parser.add_argument(
        "--max-aircraft",
        type=int,
        default=Config.MAX_AIRCRAFT_COUNT_EVAL,
        help="Maximum aircraft count for generated world.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for generated world JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    simulator = Simulator(
        Config.DEFAULT_EVALUATION_DIFFICULTY,
        get_config_path(),
        RouteSectionStore(),
        settings=settings_from_config().simulator,
    )
    world = simulator.reset(
        difficulty=Config.DEFAULT_EVALUATION_DIFFICULTY,
        create_world=True,
        seed=args.seed,
        reset_seed=True,
        eval_mode=True,
        max_aircraft_count=args.max_aircraft,
    )

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(world.json(), encoding="utf-8")

    runtime_print(_COMPONENT, "done", saved_world_json=out, aircraft=len(world.aircraft))


if __name__ == "__main__":
    main()
