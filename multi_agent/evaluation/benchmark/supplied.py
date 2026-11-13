"""The user-supplied evaluation cohort: an explicit committed seed CSV, kept in its own order.

Development accepts these seeds alongside the published validation identities; heldout and
replacement identities stay rejected even when the file lists them.
"""

import csv

from training.reservations import CONTRACT_ROOT

SUPPLIED_SEEDS = CONTRACT_ROOT / "supplied-evaluation-seeds.csv"


def supplied_seeds() -> tuple[int, ...]:
    with open(SUPPLIED_SEEDS, newline="") as stream:
        seeds = tuple(int(row["seed"]) for row in csv.DictReader(stream))
    if len(set(seeds)) != len(seeds):
        raise ValueError("supplied evaluation seeds must be duplicate-free")
    return seeds
