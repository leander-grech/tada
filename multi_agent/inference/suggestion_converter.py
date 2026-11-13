"""Convert enriched action records to TadaSuggestionMap format.

Each action record (from ATCEnv._action_history) is mapped to a frontend
suggestion dict with eventType, eventArgs, and optional timeToExecute.
"""

from collections import defaultdict
from dataclasses import dataclass

from actions.actions import Action
from config.config import Config
from inference.server_models import Airport
from simulator.route_section_store import (
    BGY_SECTION_END_MARKER,
    BGY_SECTION_START_MARKERS,
    MXP_SECTION_END_MARKER,
    MXP_SECTION_START_MARKERS,
)

# Time window (seconds) around the action step during which the command executes.
EXECUTE_TIME = 30


# ── Per-airport section configuration ────────────────────────────────────────

MXP_TROMBONE_END = "MC435"
BGY_TROMBONE_END = "ULVUN"


@dataclass(frozen=True)
class _AirportSectionConfig:
    airport: Airport
    section_start_waypoints: frozenset[str]
    section_end: str
    trombone_end: str  # apex waypoint — its presence means the full section is restored


_MXP_CONFIG = _AirportSectionConfig(
    airport=Airport.MXP,
    section_start_waypoints=frozenset(MXP_SECTION_START_MARKERS),
    section_end=MXP_SECTION_END_MARKER,
    trombone_end=MXP_TROMBONE_END,
)

_BGY_CONFIG = _AirportSectionConfig(
    airport=Airport.BGY,
    section_start_waypoints=frozenset(BGY_SECTION_START_MARKERS),
    section_end=BGY_SECTION_END_MARKER,
    trombone_end=BGY_TROMBONE_END,
)

_ALL_CONFIGS = [_MXP_CONFIG, _BGY_CONFIG]


def _detect_config(waypoints: list[str]) -> _AirportSectionConfig | None:
    """Determine airport config from route waypoint names."""
    wp_set = set(waypoints)
    for cfg in _ALL_CONFIGS:
        if cfg.section_end in wp_set or wp_set & cfg.section_start_waypoints:
            return cfg
    return None


def _math_to_aviation_heading(math_deg: float) -> float:
    """Convert math degrees (0=East, CCW) to aviation heading (0=North, CW)."""
    return round((-math_deg + 90) % 360 / 5) * 5


def _time_to_execute(decision_time_s: float, snapshot_time: float = 0.0) -> dict:
    """Build a timeToExecute dict centred on the decision time.

    Args:
        decision_time_s: Sim-relative time (seconds) at which the decision was made.
        snapshot_time: Frontend scenario time offset to convert back to absolute time.
    """
    middle = decision_time_s + snapshot_time
    start = round(middle - EXECUTE_TIME / 2)
    return {"start": start, "duration": EXECUTE_TIME}


# ── Speed actions ────────────────────────────────────────────────────────────

_SPEED_ACTIONS: set[int] = {
    Action.SLOW_DOWN_SMALL,
    Action.SLOW_DOWN_MEDIUM,
    Action.SLOW_DOWN_LARGE,
    Action.SPEED_UP_SMALL,
    Action.SPEED_UP_MEDIUM,
}


def _convert_speed(record: dict, snapshot_time: float = 0.0) -> dict | None:
    if record["action"] not in _SPEED_ACTIONS:
        return None
    speed_after = record.get("speed_after")
    if speed_after is None:
        return None
    return {
        "eventType": "aircraft-speed",
        "eventArgs": [
            record["call_sign"],
            round(speed_after / Config.SPEED_RESOLUTION) * Config.SPEED_RESOLUTION,
        ],
        "timeToExecute": _time_to_execute(record["decision_time_s"], snapshot_time),
    }


# ── Heading / turn actions ───────────────────────────────────────────────────

_TURN_ACTIONS: set[int] = {
    Action.TURN_LEFT_AND_RETURN_TO_ROUTE,
    Action.TURN_RIGHT_AND_RETURN_TO_ROUTE,
}

# turnDirection 0 = shortest path (let the frontend decide).
_TURN_DIRECTION_SHORTEST = 0


def _convert_heading(record: dict, snapshot_time: float = 0.0) -> dict | None:
    if record["action"] not in _TURN_ACTIONS:
        return None

    heading_after = record.get("heading_after")
    if heading_after is None:
        return None

    new_heading_aviation = _math_to_aviation_heading(heading_after)

    return {
        "eventType": "aircraft-heading",
        "eventArgs": [
            record["call_sign"],
            new_heading_aviation,
            _TURN_DIRECTION_SHORTEST,
        ],
        "timeToExecute": _time_to_execute(record["decision_time_s"], snapshot_time),
    }


# ── Return-to-route action ─────────────────────────────────────────────────


def _convert_return_to_route(record: dict, snapshot_time: float = 0.0) -> dict | None:
    if record["action"] != Action.RETURN_TO_ROUTE:
        return None
    return_wp = record.get("return_wp")
    if not return_wp:
        return None
    return {
        "eventType": "aircraft-waypoint",
        "eventArgs": [record["call_sign"], {"name": return_wp}],
        "timeToExecute": _time_to_execute(record["decision_time_s"], snapshot_time),
    }


# ── Skip-waypoint actions ───────────────────────────────────────────────────

_SKIP_ACTIONS: dict[int, tuple[int, int]] = {
    # (num_wps_to_skip, offset) — mirrors create_skip_waypoints_command args
    Action.SKIP_1_WAYPOINT: (1, 0),
    Action.SKIP_2_WAYPOINTS_NOT_NEXT: (2, 1),
    Action.SKIP_3_WAYPOINTS_NOT_NEXT: (3, 1),
    Action.SKIP_4_WAYPOINTS_NOT_NEXT: (4, 1),
}


def _convert_skip(record: dict, snapshot_time: float = 0.0) -> dict | None:
    params = _SKIP_ACTIONS.get(record["action"])
    if params is None:
        return None

    _, offset = params
    route_after: list[str] = record.get("route_after", [])

    if offset > 0:
        # SKIP_2/3/4: "from" is the first waypoint before the skip, "to" is the
        # first waypoint after the skipped segment in the original route.
        if len(route_after) < 2:
            return None
        return {
            "eventType": "aircraft-dct-from-to",
            "eventArgs": [
                record["call_sign"],
                {"from": route_after[0], "to": route_after[1]},
            ],
        }
    else:
        # SKIP_1: offset=0 means the first waypoint is skipped — no "from".
        # Use aircraft-waypoint with the new first waypoint.
        if route_after:
            return {
                "eventType": "aircraft-waypoint",
                "eventArgs": [record["call_sign"], {"name": route_after[0]}],
                "timeToExecute": _time_to_execute(record["decision_time_s"], snapshot_time),
            }

    return None


# ── Vector to ILS ────────────────────────────────────────────────────────────


def _convert_vector_to_ils(record: dict, snapshot_time: float = 0.0) -> dict | None:
    if record["action"] != Action.VECTOR_TO_ILS:
        return None

    route_after: list[str] = record.get("route_after", [])
    ils_waypoint = route_after[0] if route_after else None
    if not ils_waypoint:
        return None

    return {
        "eventType": "aircraft-waypoint",
        "eventArgs": [record["call_sign"], {"name": ils_waypoint}],
        "timeToExecute": _time_to_execute(record["decision_time_s"], snapshot_time),
    }


# ── Trombone actions ──────────────────────────────────────────────────

_TROMBONE_ACTIONS = {int(Action.LENGTHEN_TROMBONE), int(Action.SHORTEN_TROMBONE)}


def _is_full_section(cfg: _AirportSectionConfig, trombone_wps: list[str]) -> bool:
    return cfg.trombone_end in trombone_wps


def _convert_trombone(record: dict, _snapshot_time: float = 0.0) -> dict | None:
    """Derive a dct-from-to suggestion from the action's route_after.

    Looks at the current trombone shape — waypoints between the section start
    marker and section end marker in route_after — and builds the suggestion
    using the airport-specific rule. Works uniformly for LENGTHEN and SHORTEN.
    Returns None when the full section is present (equivalent to the bare route).
    """
    route_after: list[str] = record.get("route_after", [])

    cfg = _detect_config(route_after)
    if cfg is None:
        return None

    start_idx = next(
        (i for i, wp in enumerate(route_after) if wp in cfg.section_start_waypoints),
        None,
    )
    if start_idx is None:
        return None
    try:
        end_idx = route_after.index(cfg.section_end, start_idx + 1)
    except ValueError:
        return None

    section_start = route_after[start_idx]
    trombone_wps = route_after[start_idx + 1 : end_idx]

    if _is_full_section(cfg, trombone_wps):
        return None

    if cfg.airport == Airport.MXP and len(trombone_wps) >= 2:
        mid = len(trombone_wps) // 2
        from_wp, to_wp = trombone_wps[mid - 1], trombone_wps[mid]
    elif cfg.airport == Airport.BGY and len(trombone_wps) >= 1:
        from_wp, to_wp = trombone_wps[-1], cfg.section_end
    else:
        from_wp, to_wp = section_start, cfg.section_end

    return {
        "eventType": "aircraft-dct-from-to",
        "eventArgs": [
            record["call_sign"],
            {"from": from_wp, "to": to_wp},
        ],
    }


_ROUTE_SHORTENING_ACTIONS: set[int] = set(_SKIP_ACTIONS) | {Action.VECTOR_TO_ILS}


def _route_shortening_action_dropped_wp(
    wp: str, call_sign: str, actions_to_check: list[dict]
) -> bool:
    """True if any skip or direct-to-ILS in *actions_to_check* explicitly dropped *wp*.

    "Dropped" means the waypoint was in the action's route_before but not in
    route_after — which rules out the aircraft having passed it naturally before
    the action.
    """
    for record in actions_to_check:
        if record["call_sign"] != call_sign:
            continue
        if record.get("action", 0) not in _ROUTE_SHORTENING_ACTIONS:
            continue
        route_before = record.get("route_before")
        route_after = record.get("route_after")
        if route_before is None or route_after is None:
            continue
        if wp in route_before and wp not in route_after:
            return True
    return False


def _later_skip_dropped_trombone_from_wp(
    trombone_suggestion: dict, call_sign: str, later_actions: list[dict]
) -> bool:
    from_wp = trombone_suggestion["eventArgs"][1].get("from")
    if from_wp is None:
        return False
    return _route_shortening_action_dropped_wp(from_wp, call_sign, later_actions)


def _later_skip_same_from_as_trombone(
    trombone_suggestion: dict, call_sign: str, later_actions: list[dict]
) -> bool:
    """True if any later SKIP_2/3/4 for the same callsign has route_after[0]
    equal to the trombone suggestion's 'from' waypoint — the skip's own
    dct-from-to starts from the same waypoint as the trombone and supersedes it.
    """
    from_wp = trombone_suggestion["eventArgs"][1].get("from")
    if from_wp is None:
        return False
    for later in later_actions:
        if later["call_sign"] != call_sign:
            continue
        params = _SKIP_ACTIONS.get(later.get("action", 0))
        if params is None:
            continue
        _, offset = params
        if offset <= 0:
            continue
        route_after = later.get("route_after") or []
        if route_after and route_after[0] == from_wp:
            return True
    return False


def _superseded_by_later_route_shortening(record: dict, later_actions: list[dict]) -> bool:
    """True if a later route-shortening action for the same callsign either
    shares the same starting waypoint (route_before[0]) — the aircraft had not
    advanced between the two — or has this record's dropped waypoints back in
    its own route_before, meaning this record's route change did not persist.
    """
    route_before = record.get("route_before")
    if not route_before:
        return False
    starting_wp = route_before[0]
    dropped_wps = set(route_before) - set(record.get("route_after") or [])
    call_sign = record["call_sign"]
    for later in later_actions:
        if later["call_sign"] != call_sign:
            continue
        if later.get("action", 0) not in _ROUTE_SHORTENING_ACTIONS:
            continue
        later_route_before = later.get("route_before")
        if not later_route_before:
            continue
        if later_route_before[0] == starting_wp:
            return True
        if dropped_wps & set(later_route_before):
            return True
    return False


def _last_trombone_per_callsign(actions: list[dict]) -> set[int]:
    """Return indices of the last trombone record per callsign."""
    last_idx: dict[str, int] = {}
    for i, record in enumerate(actions):
        if record.get("action", 0) in _TROMBONE_ACTIONS:
            last_idx[record["call_sign"]] = i
    return set(last_idx.values())


# ── Main converter ───────────────────────────────────────────────────────────

_CONVERTERS = [
    (_SPEED_ACTIONS, _convert_speed),
    (_TURN_ACTIONS, _convert_heading),
    ({Action.RETURN_TO_ROUTE}, _convert_return_to_route),
    (set(_SKIP_ACTIONS), _convert_skip),
    ({Action.VECTOR_TO_ILS}, _convert_vector_to_ils),
    (_TROMBONE_ACTIONS, _convert_trombone),
]


def convert_actions_to_suggestions(
    actions: list[dict],
    all_routes: dict[str, list[str]] | None = None,
    *,
    snapshot_time: float = 0.0,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Convert a list of enriched action records into a TadaSuggestionMap.

    Returns a tuple of:
      1. suggestions — dict mapping callsign → list of suggestion dicts.
      2. raw_actions — dict mapping callsign → list of raw action dicts for
         actions that produced no frontend suggestion (e.g. full-section
         trombone restorations).

    Actions with action==0 (DO_NOTHING) are skipped.

    *all_routes* maps every callsign → remaining route waypoint names
    (from currentIndex onward). Used for trombone fallback logic.

    *snapshot_time* is the frontend's current scenario time in seconds,
    added as an offset to all suggestion timeToExecute values.
    """
    suggestions: dict[str, list[dict]] = defaultdict(list)
    raw_actions: dict[str, list[dict]] = defaultdict(list)
    last_trombones = _last_trombone_per_callsign(actions)

    for i, record in enumerate(actions):
        action_id = record.get("action", 0)
        if action_id == Action.DO_NOTHING:
            continue

        # Only process the last trombone action per callsign: trombone suggestions
        # are built from route_after, which already reflects the cumulative effect
        # of all prior trombone actions — the last action gives us the final route.
        if action_id in _TROMBONE_ACTIONS and i not in last_trombones:
            continue

        suggestion: dict | None = None
        for action_set, converter in _CONVERTERS:
            if action_id in action_set:
                suggestion = converter(record, snapshot_time)
                break

        was_suppressed = False

        # Suppress trombone suggestion if a later skip either explicitly dropped
        # its "from" waypoint or issues its own dct-from-to starting from the
        # same "from" waypoint (superseding the trombone's direct leg).
        if suggestion is not None and action_id in _TROMBONE_ACTIONS:
            later = actions[i + 1 :]
            if _later_skip_dropped_trombone_from_wp(
                suggestion, record["call_sign"], later
            ) or _later_skip_same_from_as_trombone(suggestion, record["call_sign"], later):
                suggestion = None
                was_suppressed = True

        # Suppress route-shortening suggestions where the action didn't actually
        # shorten the route (route_before == route_after). The suggestion would
        # map to a no-op command on the frontend.
        if (
            suggestion is not None
            and action_id in _ROUTE_SHORTENING_ACTIONS
            and record.get("route_before") == record.get("route_after")
        ):
            suggestion = None
            was_suppressed = True

        # Suppress route-shortening suggestion if a later route-shortening action
        # shares the same starting waypoint — the later action supersedes this one.
        if suggestion is not None and action_id in _ROUTE_SHORTENING_ACTIONS:
            if _superseded_by_later_route_shortening(record, actions[i + 1 :]):
                suggestion = None
                was_suppressed = True

        raw_action = {**record, "actionName": Action(record["action"]).name}
        if suggestion is not None:
            suggestion["rawAction"] = raw_action
            suggestions[record["call_sign"]].append(suggestion)
        elif action_id in _TROMBONE_ACTIONS or was_suppressed:
            raw_actions[record["call_sign"]].append(raw_action)
            # Ensure aircraft appears in suggestions even with an empty list
            if record["call_sign"] not in suggestions:
                suggestions[record["call_sign"]] = []

    # For aircraft with section routes but no trombone actions,
    # add a dct-from-to suggestion from section start to section end.
    if all_routes is not None:
        callsigns_with_trombone_actions = {actions[idx]["call_sign"] for idx in last_trombones}

        for callsign, route in all_routes.items():
            if callsign in callsigns_with_trombone_actions:
                continue
            cfg = _detect_config(route)
            if cfg is None:
                continue
            if cfg.section_end not in route:
                continue
            end_idx = route.index(cfg.section_end)
            if end_idx == 0:
                continue
            wp_before = next((wp for wp in route if wp in cfg.section_start_waypoints), None)
            if wp_before is None:
                continue

            # Suppress fallback if a skip or direct-to-ILS dropped the section start marker
            if _route_shortening_action_dropped_wp(wp_before, callsign, actions):
                continue

            suggestions[callsign].append(
                {
                    "eventType": "aircraft-dct-from-to",
                    "eventArgs": [callsign, {"from": wp_before, "to": cfg.section_end}],
                }
            )

    return dict(suggestions), dict(raw_actions)
