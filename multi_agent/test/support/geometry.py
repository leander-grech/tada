"""Real route points, aircraft designs and simulators for unit tests."""

from flight_simulator import MissionDifficulty, RoutePoint, Vec2, WaypointType

from atc_env.outcome_state import OutcomeState
from config.settings import settings_from_config
from config.simulation_config_path import SimulationConfigPath
from simulator.design_parser import AircraftDesignData
from simulator.route_section_store import RouteSectionStore
from simulator.simulator import Simulator


def route_point(
    name: str, x: float = 0.0, y: float = 0.0, target_speed_knots: float | None = None
) -> RoutePoint:
    return RoutePoint.create(
        waypoint_name=name,
        position_nm=Vec2(float(x), float(y)),
        waypoint_type=WaypointType.FLY_BY,
        min_altitude_ft=None,
        max_altitude_ft=None,
        target_speed_knots=target_speed_knots,
    )


def design_data() -> AircraftDesignData:
    """Aircraft that allows every speed action."""
    return AircraftDesignData(
        icao_typename="TEST",
        descriptive_name="Test Aircraft",
        max_altitude=40000.0,
        min_speed=120.0,
        max_speed=350.0,
        performance_sectors=[],
    )


def make_simulator(
    owner: OutcomeState | None = None, *, store: RouteSectionStore | None = None
) -> Simulator:
    """A real simulator over the repository scenario config, with an outcome owner."""
    return Simulator(
        MissionDifficulty.VALIDATION_USE_CASE_2,
        SimulationConfigPath("config/simulation_config"),
        store or RouteSectionStore(),
        settings=settings_from_config().simulator,
        outcome_owner=owner or OutcomeState(settings_from_config().safety),
    )
