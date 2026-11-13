"""Explicit development roles: deny heldout and replacement identities."""

from training.reservations import validate_reservation_manifest


def authorize_development(seeds, *, campaign="bgy"):
    from evaluation.benchmark.use_cases import procedure_inputs
    from evaluation.benchmark.supplied import supplied_seeds

    reservation, _ = procedure_inputs(campaign)
    validate_reservation_manifest(reservation)
    published = {row["seed"] for row in reservation["validation"]}
    protected = {row["seed"] for role in ("heldout", "replacements") for row in reservation[role]}
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("development identities must be nonempty, integer and duplicate-free")
    if not set(seeds) <= published | set(supplied_seeds()) or set(seeds) & protected:
        raise ValueError(
            "development access denies nonpublished, heldout and replacement identities"
        )
