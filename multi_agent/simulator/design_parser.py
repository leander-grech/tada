"""This module contains the code for parsing the aircraft design data."""

from typing import Any, NamedTuple


class PerformanceSectorData(NamedTuple):
    """Performance data for a specific altitude sector."""

    altitude_min: float
    altitude_max: float
    optimal_cruise_speed: float
    optimal_climb_speed: float
    optimal_descent_speed: float
    vertical_climb_speed: float
    vertical_descent_speed: float


class AircraftDesignData(NamedTuple):
    """Parsed aircraft design data."""

    icao_typename: str
    descriptive_name: str
    max_altitude: float
    min_speed: float
    max_speed: float
    performance_sectors: list[PerformanceSectorData]


def parse_aircraft_design(design: Any) -> AircraftDesignData:
    """Parse an aircraft design object into a structured format."""
    max_altitude: float = design.limits[1]
    min_speed: float = design.limits[3]
    max_speed: float = design.limits[4]

    performance_sectors: list[PerformanceSectorData] = []

    for sector in design.performance_sectors:
        cruise_speed: float = sector.cruise.v_opt_kcas
        climb_speed: float = sector.climb.v_opt_kcas
        descent_speed: float = sector.descent.v_opt_kcas

        climb_v_speed: float = (
            sector.climb.vertical_speed_fpm if sector.climb.vertical_speed_fpm is not None else 0.0
        )
        descent_v_speed: float = (
            sector.descent.vertical_speed_fpm
            if sector.descent.vertical_speed_fpm is not None
            else 0.0
        )

        performance_sectors.append(
            PerformanceSectorData(
                altitude_min=sector.altitude_min_ft,
                altitude_max=sector.altitude_max_ft,
                optimal_cruise_speed=cruise_speed,
                optimal_climb_speed=climb_speed,
                optimal_descent_speed=descent_speed,
                vertical_climb_speed=climb_v_speed,
                vertical_descent_speed=descent_v_speed,
            )
        )

    return AircraftDesignData(
        icao_typename=design.icao_typename,
        descriptive_name=design.descriptive_name,
        max_altitude=max_altitude,
        min_speed=min_speed,
        max_speed=max_speed,
        performance_sectors=performance_sectors,
    )


def parse_all_aircraft_designs(designs: dict[str, Any]) -> dict[str, AircraftDesignData]:
    """
    Parse all aircraft designs from the designs dictionary.
    Returns a dictionary mapping each design key to its parsed parameters.
    """
    return {key: parse_aircraft_design(design) for key, design in designs.items()}
