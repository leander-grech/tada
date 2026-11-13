"""Materialize development manifests for the user-supplied seed cohort, in the file's own order.

No policy is stepped and no outcome is opened; existing manifests are never overwritten.
"""

import argparse
from pathlib import Path

from common.runtime_log import runtime_print
from evaluation.benchmark.contract import materialize_manifest, procedure_contract
from evaluation.benchmark.generator_variant import VARIANT
from evaluation.benchmark.supplied import supplied_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="manifest directory")
    parser.add_argument("--procedure", choices=("mxp", "bgy"), nargs="+", default=["mxp", "bgy"])
    parser.add_argument("--aircraft", type=int, nargs="+", default=[20, 15, 10])
    parser.add_argument("--generator-variant", choices=[VARIANT])
    args = parser.parse_args()
    seeds = list(supplied_seeds())
    for campaign in args.procedure:
        for count in args.aircraft:
            contract = procedure_contract(
                "development",
                campaign=campaign,
                aircraft_count=count,
                generator_variant=args.generator_variant,
            )
            path = args.output / f"{campaign}-development-{count}.json"
            runtime_print(
                "MaterializeSupplied",
                "written",
                path=str(materialize_manifest(contract, seeds, path)),
            )


if __name__ == "__main__":
    main()
