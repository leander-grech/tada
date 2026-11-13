"""Converts frontend SerializedAircraft[] snapshots into World JSON for inference."""

import json
import logging
import math
from pathlib import Path
from typing import Any

from common.jsonio import read_json
from config.config import Config
from config.simulation_config_path import get_config_path
from inference.server_models import BGY_PREFERRED_ALTITUDE_NUM_LEVELS, Airport

logger = logging.getLogger(__name__)

# Unit conversion constants (same as actions/actions.py)
NM_TO_M = 1852
KTS_TO_MS = 0.5144
FT_TO_M = 0.3048

# Airport / runway constants
MXP_RUNWAY_ID = "35L"
MXP_FINAL_FIX = [-43003.76, 18865.87]
BGY_RUNWAY_ID = "10"
BGY_FINAL_FIX = [34869.94, 25137.05]

# Aircraft status enum values
STATUS_FINISHED = 3


def load_design_block(config_root: str | None = None) -> dict[str, Any]:
    """Serialize the locally supplied aircraft designs into the World JSON form.

    Aircraft performance data is not distributed with the repository; it is read from
    ``<config_root>/designs/*.json`` and converted by the native loader.
    """
    import flight_simulator as fs

    difficulty = Config.DEFAULT_EVALUATION_DIFFICULTY
    config = fs.load_config(config_root or get_config_path().simulation_config_root, difficulty)
    if not config.aircraft_designs:
        raise FileNotFoundError("No aircraft designs found; see the README prerequisites")
    mission = fs.Mission(
        completed=False,
        failed=False,
        aircraft_horizontal_safezone_radius_nm=5,
        aircraft_vertical_safezone_radius_ft=1000,
    )
    _, waypoints, airspace, *_ = fs.generate_random_mission(
        1, config, mission, False, None, difficulty
    )
    world = fs.load_world([], None, config.aircraft_designs, airspace, waypoints)
    return json.loads(world.json())["designs"]


class InvalidPreferredAltitudeError(Exception):
    """Aircraft preferredAltitudeLevel exceeds the effective num_levels."""


class SnapshotConverter:
    """Converts frontend aircraft snapshots to World JSON suitable for inference.

    Loads a template world-base.json and the locally supplied aircraft designs at
    init, then produces a complete World JSON string from a list of SerializedAircraft
    dicts.
    """

    def __init__(self) -> None:
        template_path = Path(__file__).parent / "world-base.json"
        self._template: dict[str, Any] = read_json(template_path)
        self._designs: dict[str, Any] = load_design_block()

        self._waypoints: dict[str, dict] = self._template.get("waypoints", {})

    def convert(
        self,
        aircraft_list: list[dict[str, Any]],
        *,
        snapshot_time: float = 0.0,
        preferred_altitude_num_levels: int = BGY_PREFERRED_ALTITUDE_NUM_LEVELS,
    ) -> tuple[str, dict[str, str]]:
        """Convert a list of SerializedAircraft dicts to a World JSON string.

        Args:
            aircraft_list: Frontend SerializedAircraft dicts.
            snapshot_time: The frontend's current scenario time in seconds.
                Used to convert absolute amanArrivalTime to sim-relative eta_s.
            preferred_altitude_num_levels: Number of preferred-altitude levels
                used to validate and encode preferredAltitudeLevel for BGY
                aircraft.

        Returns:
            (world_json_str, callsign_map) where callsign_map maps original
            callsigns to remapped TESTnnn callsigns.
        """
        callsign_map: dict[str, str] = {}
        converted_aircraft: list[dict[str, Any]] = []
        arrival_targets: list[dict[str, Any]] = []
        agent_index = 1
        non_finished_count = 0

        for ac in aircraft_list:
            status = ac["state"]["status"]
            if status == STATUS_FINISHED:
                continue
            non_finished_count += 1

            original_callsign = ac["callSign"]
            if original_callsign in callsign_map:
                logger.warning(
                    "Skipping duplicate aircraft callsign '%s' in snapshot input",
                    original_callsign,
                )
                continue
            remapped_callsign = f"TEST{agent_index:03d}"

            try:
                converted = self._convert_aircraft(
                    ac, remapped_callsign, preferred_altitude_num_levels
                )
            except ValueError as exc:
                logger.warning("Skipping aircraft '%s': %s", original_callsign, exc)
                continue

            callsign_map[original_callsign] = remapped_callsign
            agent_index += 1
            converted_aircraft.append(converted)

            aman_time = ac.get("amanArrivalTime")
            if aman_time is not None:
                arrival_targets.append(
                    {
                        "ac_callsign": remapped_callsign,
                        "eta_s": aman_time - snapshot_time,
                    }
                )

        if non_finished_count > 0 and not converted_aircraft:
            raise ValueError(
                "Snapshot conversion produced no valid aircraft; "
                "check route data and airport waypoints."
            )

        world = {
            "aircraft": converted_aircraft,
            "designs": self._designs,
            "airspace": self._template["airspace"],
            "waypoints": self._template["waypoints"],
            "routes": {},
            "runways": {},
            "infringements": [],
            "arrival_targets": arrival_targets,
            "age_in_steps": 0,
            "age_in_seconds": 0.0,
        }

        return json.dumps(world), callsign_map

    def _convert_aircraft(
        self, ac: dict[str, Any], callsign: str, preferred_altitude_num_levels: int
    ) -> dict[str, Any]:
        """Convert a single SerializedAircraft dict to World aircraft format."""
        state = ac["state"]
        target = ac["targetState"]
        following_route = state["controlMode"] == "route"

        # Position: [x_nm, y_nm] + altitude_ft → [x_m, y_m, z_m]
        pos = state["position"]
        position_m = [
            pos[0] * NM_TO_M,
            -pos[1] * NM_TO_M,
            state["altitude"] * FT_TO_M,
        ]

        # Velocity: aviation heading + speed (kts) → [vx, vy, 0] m/s
        velocity_m_s = _aviation_to_velocity(state["heading"], state["speed"])

        # Target state: always forward speed/altitude so Rust has deceleration
        # and altitude-capture pressure that matches TS. Heading is derived from
        # route[0] by Rust when following_route is true, so it's left null.
        target_state: dict[str, float | None]
        if following_route:
            target_state = {
                "heading": None,
                "speed": target["speed"] * KTS_TO_MS,
                "altitude": target["altitude"] * FT_TO_M,
            }
        else:
            target_heading_rad = _aviation_heading_to_math_radians(target["heading"])
            target_state = {
                "heading": target_heading_rad,
                "speed": target["speed"] * KTS_TO_MS,
                "altitude": target["altitude"] * FT_TO_M,
            }

        # Route
        route, runway_id, final_fix_m = self._build_route(ac["routeTracker"])

        is_aman_ttg = (
            ac.get("amanOffsetSeconds", 0) < 0 if ac.get("amanOffsetSeconds") is not None else False
        )

        if is_aman_ttg:
            time_to_gain = -ac["amanOffsetSeconds"]
            time_to_lose = 0.0
        else:
            time_to_gain = 0.0
            time_to_lose = ac["amanOffsetSeconds"]

        return {
            "call_sign": callsign,
            "start_time_s": max(0.0, ac["secondsToStart"]),
            "under_control": False,
            "inside_airspace": True,
            "following_route": following_route,
            "arrival_checked": False,
            "aircraft_type": "A320",
            "route": route,
            "route_name": "SNAPSHOT",
            "runway_id": runway_id,
            "mission": {
                "completed": False,
                "failed": False,
                "aircraft_horizontal_safezone_radius": Config.HORIZONTAL_SAFEZONE_RADIUS_NM
                * NM_TO_M,
                "aircraft_vertical_safezone_radius": Config.VERTICAL_SAFEZONE_RADIUS_FT * FT_TO_M,
            },
            "final_fix_m": final_fix_m,
            "size_scale": 1.0,
            "manually_updated_speed": ac["manuallyUpdatedSpeed"],
            "state": {
                "position_m": position_m,
                "velocity_m_s": velocity_m_s,
            },
            "target_state": target_state,
            "hold_state": None,
            "aman": {"time_to_gain": time_to_gain, "time_to_lose": time_to_lose},
            "preferred_altitude_fraction": self._build_preferred_altitude(
                ac, preferred_altitude_num_levels
            ),
        }

    def _build_route(
        self, route_tracker: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, list[float]]:
        """Build the route waypoint list from routeTracker.

        Returns:
            (route_points, runway_id, final_fix_m)
        """
        full_route = route_tracker["route"]
        if not full_route:
            raise ValueError("Empty route — cannot build a valid route")

        current_index = int(route_tracker["currentIndex"])
        current_index = min(max(current_index, 0), len(full_route) - 1)
        remaining = full_route[current_index:]

        # Detect airport from route waypoints
        runway_id, final_fix_m = self._detect_airport(full_route)

        route_points: list[dict[str, Any]] = []
        for wp in remaining:
            name = wp["name"]
            min_alt = wp.get("min_altitude")
            max_alt = wp.get("max_altitude")
            speed = wp.get("speed")
            route_points.append(
                {
                    "waypoint_name": name,
                    "position_m": None,
                    "waypoint_type": None,
                    "min_altitude": min_alt * FT_TO_M if min_alt is not None else None,
                    "max_altitude": max_alt * FT_TO_M if max_alt is not None else None,
                    "target_speed": speed * KTS_TO_MS if speed is not None else None,
                    "hold_plan": None,
                }
            )

        if not route_points:
            raise ValueError(f"Could not build route points from remaining route: {remaining}")

        return route_points, runway_id, final_fix_m

    @staticmethod
    def _build_preferred_altitude(ac: dict[str, Any], num_levels: int) -> dict[str, int] | None:
        """Build preferred_altitude_fraction for the World JSON aircraft.

        Uses the frontend-supplied preferredAltitudeLevel for BGY aircraft.
        Raises InvalidPreferredAltitudeError when the level is out of range for
        the effective num_levels (levels are 0-indexed: 0..num_levels-1).
        """
        level = ac.get("preferredAltitudeLevel")
        if level is None:
            return None
        if level >= num_levels:
            raise InvalidPreferredAltitudeError(
                f"Aircraft {ac.get('callSign')!r} has preferredAltitudeLevel={level} "
                f"which is out of range for num_levels={num_levels}"
            )
        return {
            "preferred_level": level,
            "num_levels": num_levels,
        }

    _AIRPORT_RUNWAY_MAP: dict[Airport, tuple[str, list[float]]] = {
        Airport.MXP: (MXP_RUNWAY_ID, MXP_FINAL_FIX),
        Airport.BGY: (BGY_RUNWAY_ID, BGY_FINAL_FIX),
    }

    @staticmethod
    def _detect_airport(
        route: list[dict[str, Any]],
    ) -> tuple[str, list[float]]:
        """Determine which airport (MXP or BGY) a route targets.

        Returns:
            (runway_id, final_fix_m)

        Raises:
            ValueError: If route is empty or has no waypoint with known airport prefix.
        """
        code = SnapshotConverter._detect_airport_code(route)
        if code is None:
            raise ValueError("Route has no waypoint that maps to airport 'MXP' or 'BGY'")
        return SnapshotConverter._AIRPORT_RUNWAY_MAP[code]

    @staticmethod
    def _detect_airport_code(route: list[dict[str, Any]]) -> Airport | None:
        """Return the Airport enum for a route, or None."""
        if not route:
            return None
        for wp in reversed(route):
            airport = SnapshotConverter._airport_from_waypoint_name(wp["name"])
            if airport is not None:
                return airport
        return None

    def convert_split(
        self,
        aircraft_list: list[dict[str, Any]],
        *,
        snapshot_time: float = 0.0,
        preferred_altitude_num_levels: int = BGY_PREFERRED_ALTITUDE_NUM_LEVELS,
    ) -> dict[Airport, tuple[str, dict[str, str]]]:
        """Split aircraft by airport and convert each group independently.

        Args:
            aircraft_list: Frontend SerializedAircraft dicts.
            snapshot_time: Frontend's current scenario time in seconds.
            preferred_altitude_num_levels: Number of preferred-altitude levels
                used to validate and encode preferredAltitudeLevel for BGY
                aircraft.

        Returns:
            Dict mapping Airport -> (world_json_str, callsign_map).
            Aircraft that are finished or whose airport cannot be detected are skipped.
        """
        groups: dict[Airport, list[dict[str, Any]]] = {}

        for ac in aircraft_list:
            if ac["state"]["status"] == STATUS_FINISHED:
                continue

            route = ac.get("routeTracker", {}).get("route", [])
            airport = self._detect_airport_code(route)
            if airport is None:
                logger.warning(
                    "Cannot detect airport for aircraft '%s', skipping", ac.get("callSign")
                )
                continue

            groups.setdefault(airport, []).append(ac)

        result: dict[Airport, tuple[str, dict[str, str]]] = {}
        for airport_code, group in groups.items():
            world_json, callsign_map = self.convert(
                group,
                snapshot_time=snapshot_time,
                preferred_altitude_num_levels=preferred_altitude_num_levels,
            )
            result[airport_code] = (world_json, callsign_map)

        return result

    @staticmethod
    def _airport_from_waypoint_name(waypoint_name: str) -> Airport | None:
        if waypoint_name == "MXP" or waypoint_name.startswith("MXP_"):
            return Airport.MXP
        if waypoint_name == "BGY" or waypoint_name.startswith("BGY_"):
            return Airport.BGY
        return None


def _aviation_heading_to_math_radians(heading_deg: float) -> float:
    """Convert aviation heading (0=N, CW) to math angle in radians."""
    math_deg = (-heading_deg + 90) % 360
    return math.radians(math_deg)


def _aviation_to_velocity(heading_deg: float, speed_kts: float) -> list[float]:
    """Convert aviation heading + speed (kts) to [vx, vy, 0] in m/s."""
    math_deg = (-heading_deg + 90) % 360
    speed_ms = speed_kts * KTS_TO_MS
    vx = speed_ms * math.cos(math.radians(math_deg))
    vy = speed_ms * math.sin(math.radians(math_deg))
    return [vx, vy, 0.0]
