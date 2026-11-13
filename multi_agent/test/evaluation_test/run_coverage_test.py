"""Tests for the plain evaluation path's manifest-coverage guarantees.

Rates are computed over the scenarios actually collected, so a walk that
skipped manifest seeds silently changes the denominator.  These tests pin the
shortfall report that gates the summary and the recovery of terminals RLlib
files as reset observations.
"""

from evaluation.run import coverage_shortfall

_MANIFEST = [101, 202, 303, 404]


def test_shortfall_lists_missing_seeds_in_manifest_order() -> None:
    assert coverage_shortfall([404, 202], _MANIFEST) == [101, 303]


def test_full_coverage_reports_no_shortfall() -> None:
    assert coverage_shortfall(reversed(_MANIFEST), _MANIFEST) == []


def test_extra_and_repeated_seeds_do_not_hide_a_shortfall() -> None:
    """Randomize-quota draws and replayed scenarios are not manifest coverage."""
    assert coverage_shortfall([101, 101, 202, 999, 999], _MANIFEST) == [303, 404]


def test_empty_collection_reports_the_whole_manifest() -> None:
    assert coverage_shortfall([], _MANIFEST) == _MANIFEST
