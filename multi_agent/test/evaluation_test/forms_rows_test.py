"""Form action counts: auto-solve extensions join the extend column, per aircraft and per episode."""

from actions.action_table import Action
from evaluation.forms.rows import (
    AIRCRAFT_HEADER,
    EPISODE_HEADER,
    LEAD,
    TMB_ACTIONS,
    aircraft_rows,
    episode_row,
)

EXTEND, SLOW = int(Action.LENGTHEN_TROMBONE), int(Action.SLOW_DOWN_SMALL)
TRACE = {
    "seed": 7,
    "mode": "deterministic",
    "ordinal": 0,
    "outcome": {"kind": "success"},
    "separation_events": 0,
    "warning_pairs": [],
    "targets": [{"callsign": "A", "eta_s": 100.0}, {"callsign": "B", "eta_s": 200.0}],
    "initial_landing_s": {"A": 90.0, "B": 230.0},
    "final_landing_s": {"A": 101.0, "B": 199.0},
    "policy_actions": {"A:0": 4, f"A:{EXTEND}": 1, "B:0": 3, f"B:{SLOW}": 2},
    "auto_solve_actions": {f"A:{EXTEND}": 1},
}
COLUMNS = [name for name, _ in TMB_ACTIONS]
AIRCRAFT_TMB = AIRCRAFT_HEADER.index(f"TMB: {COLUMNS[0]}")


def _tmb_block(row, start):
    return dict(zip(COLUMNS, row[start : start + len(COLUMNS)], strict=True))


def test_auto_solve_extensions_count_as_extensions_of_their_aircraft_and_episode():
    a, b = (_tmb_block(r, AIRCRAFT_TMB) for r in aircraft_rows(TRACE, "mxp"))
    episode = _tmb_block(episode_row(TRACE, "mxp"), len(LEAD))
    assert a["Extend trombone"] == 2 and b["Slow down"] == 2
    assert episode == {name: a[name] + b[name] for name in COLUMNS}
    assert "Auto-solve extend" not in EPISODE_HEADER[2]
    clearances = AIRCRAFT_HEADER.index("Number of non-'do nothing' clearances")
    assert [r[clearances] for r in aircraft_rows(TRACE, "mxp")] == [2, 2]


def test_a_procedure_fills_only_its_own_action_block():
    row = next(aircraft_rows(TRACE, "pms"))
    pms = AIRCRAFT_HEADER.index("PMS: Extend PMS")
    assert row[AIRCRAFT_TMB : AIRCRAFT_TMB + len(COLUMNS)] == [""] * len(COLUMNS)
    assert row[pms] == 2
    assert len(row) == len(AIRCRAFT_HEADER)
