"""Form rows (template layout of the partner's 20-flight evaluation workbook) from replay traces.

Definitions:
- aircraft index: position in target-ETA order;
- TTL/TTG = target ETA - forecast landing with no commands from the pre-auto-solve
  world (positive: early, time to lose); initial deviation is its absolute value;
- final deviation: |landing - target ETA| of the evaluated trajectory (blank if the
  aircraft never landed);
- AMAN sequence: landing position equals target-ETA position (same-second ties keep
  target order);
- clearances and action counts: policy decisions plus the pre-episode auto-solve
  extensions, which are trombone extensions and count in the extend column (the
  internal return-to-route marker is excluded).
"""

from collections import Counter, defaultdict

from actions.action_table import Action

PROCEDURE = {"mxp": "TMB", "pms": "PMS"}

_AIRCRAFT_LEAD = [
    "Agent Type (Multi|Single)",
    "Procedure type (PMS|TMB)",
    "Seed number (100 seeds)",
    "Type of policy (Deterministicx1|Stochasticx20) 0-20",
    "Aircraft Index (0-19)",
    "TTL or TTG value (seconds)\n+ve for TTL and -ve for TTG ",
    "Initial absolute time deviation (seconds)",
    "Final absolute time deviation (seconds)",
    "Landed according to AMAN sequence? (Yes | No)",
    "Number of non-'do nothing' clearances",
    # Appended columns (not in the template).
    "Call sign",
    "Target ETA (s)",
    "Final landing time (s)",
]

SLOW = (Action.SLOW_DOWN_SMALL, Action.SLOW_DOWN_MEDIUM, Action.SLOW_DOWN_LARGE)
SPEED = (Action.SPEED_UP_SMALL, Action.SPEED_UP_MEDIUM)
TURN = (Action.TURN_LEFT_AND_RETURN_TO_ROUTE, Action.TURN_RIGHT_AND_RETURN_TO_ROUTE)
SKIP = (
    Action.SKIP_1_WAYPOINT,
    Action.SKIP_2_WAYPOINTS_NOT_NEXT,
    Action.SKIP_3_WAYPOINTS_NOT_NEXT,
    Action.SKIP_4_WAYPOINTS_NOT_NEXT,
)
# Template action columns per procedure. On PMS the extend/shorten pair acts on the point-merge
# leg, and the vector to the ILS leaves it for the merge point.
TMB_ACTIONS = [
    ("Do nothing", (Action.DO_NOTHING,)),
    ("Slow down", SLOW),
    ("Speed up", SPEED),
    ("Turn and return to route", TURN),
    ("Extend trombone", (Action.LENGTHEN_TROMBONE,)),
    ("Retract trombone", (Action.SHORTEN_TROMBONE,)),
    ("Skip waypoints", SKIP),
    ("Vector to ILS", (Action.VECTOR_TO_ILS,)),
]
PMS_ACTIONS = [
    ("Do nothing", (Action.DO_NOTHING,)),
    ("Slow down", SLOW),
    ("Speed up", SPEED),
    ("Turn and return to route", TURN),
    ("Extend PMS", (Action.LENGTHEN_TROMBONE,)),
    ("Retract PMS", (Action.SHORTEN_TROMBONE,)),
    ("Skip waypoints", SKIP),
    ("Direct to merge point", (Action.VECTOR_TO_ILS,)),
]
LEAD = [
    "Agent Type (Multi|Single)",
    "Procedure type (PMS|TMB)",
    "Seed number (100 seeds)",
    "Type of policy (Deterministicx1|Stochasticx20) 0-20",
    "End reason (success|delayed|timeout|airspace_exit|Loss of separation)",
    "Total initial absolute time deviation (seconds)",
    "Total final absolute time deviation (seconds)",
    "AMAN sequence met? (Yes|No)",
    "Infringements (not conflicts) (Yes|No)",
]
# Appended per-aircraft action columns (not in the template), in the per-episode layout.
AIRCRAFT_HEADER = [
    *_AIRCRAFT_LEAD,
    *(f"TMB: {name}" for name, _ in TMB_ACTIONS),
    *(f"PMS: {name}" for name, _ in PMS_ACTIONS),
]
_BLANK = [""] * len(LEAD)
_GAP = [""] * (len(TMB_ACTIONS) + len(PMS_ACTIONS) - 1)
EPISODE_HEADER = [
    [*LEAD, "Number of actions", *_GAP],
    [*_BLANK, "TMB", *[""] * (len(TMB_ACTIONS) - 1), "PMS", *[""] * (len(PMS_ACTIONS) - 1)],
    [*_BLANK, *(name for name, _ in TMB_ACTIONS), *(name for name, _ in PMS_ACTIONS)],
]

END_REASON = {
    "verified_noop_success": "success",
    "success": "success",
    "safe_delayed": "delayed",
    # Unresolved attempts are budget/verification exhaustion without a safety event.
    "unresolved": "timeout",
}


def end_reason(trace) -> str:
    kind = trace["outcome"]["kind"]
    if kind == "infringement":
        return "Loss of separation" if trace["separation_events"] else "airspace_exit"
    return END_REASON[kind]


def policy_index(trace) -> int:
    """Greedy 0, stochastic ordinal i column i + 1, search attempt a column a."""
    return trace["ordinal"] + (trace["mode"] == "stochastic")


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


def seconds(value) -> str:
    return "" if value is None else f"{value:.2f}"


def aircraft_timing(trace):
    """Per-aircraft (callsign, eta, ttl, final landing, in-sequence) in target-ETA order."""
    targets = sorted(trace["targets"], key=lambda t: t["eta_s"])
    order = {t["callsign"]: i for i, t in enumerate(targets)}
    final = trace["final_landing_s"]
    landed = sorted(final, key=lambda cs: (final[cs], order[cs]))
    position = {cs: i for i, cs in enumerate(landed)}
    for t in targets:
        cs = t["callsign"]
        initial = trace["initial_landing_s"].get(cs)
        yield {
            "callsign": cs,
            "eta": t["eta_s"],
            "ttl": None if initial is None else t["eta_s"] - initial,
            "landing": final.get(cs),
            "in_sequence": position.get(cs) == order[cs],
        }


def action_counts(trace) -> dict[str, Counter]:
    """Per-callsign action counts: policy decisions plus pre-episode auto-solve extensions."""
    counts = defaultdict(Counter)
    for key, count in (*trace["policy_actions"].items(), *trace["auto_solve_actions"].items()):
        callsign, action = key.rsplit(":", 1)
        counts[callsign][int(action)] += count
    return counts


def _block(per_action, columns):
    return [sum(per_action[a] for a in actions) for _, actions in columns]


def _blocks(per_action, procedure):
    """The TMB and PMS action columns; only the procedure's own block is filled."""
    tmb = PROCEDURE[procedure] == "TMB"
    return [
        *(_block(per_action, TMB_ACTIONS) if tmb else [""] * len(TMB_ACTIONS)),
        *([""] * len(PMS_ACTIONS) if tmb else _block(per_action, PMS_ACTIONS)),
    ]


def aircraft_rows(trace, procedure):
    counts = action_counts(trace)
    for index, row in enumerate(aircraft_timing(trace)):
        ttl = row["ttl"]
        per_action = counts[row["callsign"]]
        yield [
            "Multi",
            PROCEDURE[procedure],
            trace["seed"],
            policy_index(trace),
            index,
            seconds(ttl),
            seconds(None if ttl is None else abs(ttl)),
            seconds(None if row["landing"] is None else abs(row["landing"] - row["eta"])),
            yes_no(row["in_sequence"]),
            per_action.total() - per_action[Action.DO_NOTHING],
            row["callsign"],
            seconds(row["eta"]),
            seconds(row["landing"]),
            *_blocks(per_action, procedure),
        ]


def episode_row(trace, procedure):
    timing = list(aircraft_timing(trace))
    per_action = sum(action_counts(trace).values(), Counter())
    initial = sum(abs(r["ttl"]) for r in timing if r["ttl"] is not None)
    final = sum(abs(r["landing"] - r["eta"]) for r in timing if r["landing"] is not None)
    return [
        "Multi",
        PROCEDURE[procedure],
        trace["seed"],
        policy_index(trace),
        end_reason(trace),
        seconds(initial),
        seconds(final),
        yes_no(all(r["in_sequence"] for r in timing)),
        yes_no(bool(trace["warning_pairs"])),
        *_blocks(per_action, procedure),
    ]
