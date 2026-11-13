"""Per-aircraft scalar and flight-plan (waypoint) observation features."""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from flight_simulator import Vec3

from config.settings import ObservationScaleSettings
from geometry.vectoring import resolve_waypoint_position
from models.feature_transforms import (
    _bearing_sin_cos,
    _distance,
    _heading_rad,
    bounded_signed,
    map_sym,
    map_to_sym_unit,
    representative_altitude,
)
from models.observation_settings import ObservationSettings


@dataclass
class AircraftScalarFeatures:
    """Container for aircraft-level scalar features with normalization capabilities."""

    vel_xy: float  # Velocity on xy plane (kts)
    vz: float  # Vertical velocity (fpm)
    heading_sin: float  # Sine of current heading
    heading_cos: float  # Cosine of current heading
    time_to_target: float  # Time to target (s), soft-bounded mapping
    deviation_proxy: float  # Forecast schedule-deviation proxy (s)
    under_control: float  # Under ATC control (0/1)
    time_since_last_action: float  # Time since last non-zero action (s)
    timing_available: float = 1.0

    def to_numpy(
        self, settings: ObservationSettings, scales: ObservationScaleSettings
    ) -> np.ndarray:
        """Convert to normalized numpy array for neural network input."""
        return np.array(
            [
                map_to_sym_unit(self.vel_xy, 0.0, scales.vel_xy_max),
                map_sym(self.vz, scales.vel_z_max),
                float(self.heading_sin),
                float(self.heading_cos),
                bounded_signed(self.time_to_target, scales.time_to_target_max, gain=1.5),
                settings.encode_deviation(self.deviation_proxy) if self.timing_available else 0.0,
                float(self.under_control),
                map_to_sym_unit(
                    self.time_since_last_action,
                    0.0,
                    scales.time_since_last_action_max,
                ),
                float(self.timing_available),
            ],
            dtype=np.float32,
        )


@dataclass
class WaypointRelFeatures:
    """Relative waypoint features for flight-plan rows."""

    dx: float  # x offset to waypoint (nm)
    dy: float  # y offset to waypoint (nm)
    dz: float  # altitude offset to waypoint (ft)
    dist: float  # horizontal distance to waypoint (nm)
    bearing_sin: float  # sine of bearing to waypoint
    bearing_cos: float  # cosine of bearing to waypoint
    vel_target: float  # target velocity at waypoint (kts)
    vel_target_available: float  # is target velocity available (0/1)

    def to_numpy(self, scales: ObservationScaleSettings) -> np.ndarray:
        """Convert to normalized numpy array for neural network input."""
        geom = _normalize_waypoint_geometry(
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            dist=self.dist,
            bearing_sin=self.bearing_sin,
            bearing_cos=self.bearing_cos,
            scales=scales,
        )
        out = np.empty((scales.flight_plan_waypoint_dim,), dtype=np.float32)
        geom_dim = int(geom.shape[0])
        out[:geom_dim] = geom
        out[geom_dim] = (
            map_to_sym_unit(self.vel_target, 0.0, scales.vel_xy_max)
            if self.vel_target_available > 0
            else 0.0
        )
        out[geom_dim + 1] = self.vel_target_available
        return out


def _normalize_waypoint_geometry(
    dx: float,
    dy: float,
    dz: float,
    dist: float,
    bearing_sin: float,
    bearing_cos: float,
    scales: ObservationScaleSettings,
) -> np.ndarray:
    """Normalize shared geometric waypoint channels for flight-plan rows."""
    pos_xy_half_range = float(abs(scales.rel_pos_xy_max))
    dz_half_range = float(scales.rel_pos_z_range_ft)
    dist_clip_nm = float(abs(scales.rel_pos_xy_max))

    return np.array(
        [
            map_sym(dx, pos_xy_half_range),
            map_sym(dy, pos_xy_half_range),
            map_sym(dz, dz_half_range),
            map_to_sym_unit(dist, 0.0, dist_clip_nm),
            bearing_sin,
            bearing_cos,
        ],
        dtype=np.float32,
    )


def _compute_aircraft_scalar(
    vel: Vec3,
    ttt: float,
    deviation_proxy: float,
    under_control: bool,
    time_since_last_action: float,
    timing_available: bool = True,
) -> AircraftScalarFeatures:
    """Compute aircraft-level scalar features as a readable datamodel."""
    vel_xy = float(math.hypot(float(vel.x), float(vel.y)))
    heading = _heading_rad(float(vel.x), float(vel.y))
    h_sin, h_cos = math.sin(heading), math.cos(heading)

    vz = float(vel.z)

    return AircraftScalarFeatures(
        vel_xy=vel_xy,
        vz=vz,
        heading_sin=h_sin,
        heading_cos=h_cos,
        time_to_target=ttt,
        deviation_proxy=deviation_proxy,
        under_control=float(under_control),
        time_since_last_action=time_since_last_action,
        timing_available=float(timing_available),
    )


def _collect_waypoints_and_rel_features(
    aircraft, simulator, current_pos, settings: ObservationSettings
) -> tuple[list[tuple[float, float, float]], list[WaypointRelFeatures]]:
    """Resolve the full aligned physical route; bound only observation feature rows."""

    waypoints_glb: list[tuple[float, float, float]] = []
    rel_waypoints: list[WaypointRelFeatures] = []

    if aircraft.route:
        for index, rp in enumerate(aircraft.route):
            pos = resolve_waypoint_position(rp, simulator.world.waypoints)
            if pos is None:
                raise ValueError(
                    f"unsupported route for {aircraft.call_sign}: unresolved position "
                    f"at index {index} ({rp.waypoint_name})"
                )
            wp_x, wp_y = pos
            alt = representative_altitude(
                rp.min_altitude_ft, rp.max_altitude_ft, float(current_pos.z)
            )
            glb_wp = (wp_x, wp_y, float(alt))
            waypoints_glb.append(glb_wp)
            if index < settings.max_waypoints:
                rel_waypoints.append(_wp_rel_from_glb(glb_wp, current_pos, rp.target_speed))

    return waypoints_glb, rel_waypoints


def _wp_rel_from_glb(
    glb_wp: tuple[float, float, float], current_pos, vel_target: float | None
) -> WaypointRelFeatures:
    wx, wy, wz = glb_wp
    dx = wx - float(current_pos.x)
    dy = wy - float(current_pos.y)
    dz = wz - float(current_pos.z)
    dist = _distance(dx, dy)
    b_sin, b_cos = _bearing_sin_cos(dx, dy)
    vel_target = float(vel_target) if vel_target is not None else 0.0
    return WaypointRelFeatures(
        dx=dx,
        dy=dy,
        dz=dz,
        dist=dist,
        bearing_sin=b_sin,
        bearing_cos=b_cos,
        vel_target=vel_target,
        vel_target_available=1.0 if vel_target > 0.0 else 0.0,
    )


def _compute_flight_plans_rel(
    rel_waypoints: Sequence[WaypointRelFeatures],
    settings: ObservationSettings,
    scales: ObservationScaleSettings,
):
    """Pack precomputed relative waypoint features up to MAX_WAYPOINTS."""
    wp_dim = settings.waypoint_dim
    flight_plans_rel = np.zeros((settings.max_waypoints, wp_dim), dtype=np.float32)
    flight_plan_mask = np.zeros((settings.max_waypoints,), dtype=np.float32)

    for i, rel_features in enumerate(rel_waypoints[: settings.max_waypoints]):
        flight_plans_rel[i] = rel_features.to_numpy(scales)
        flight_plan_mask[i] = 1.0

    return flight_plans_rel, flight_plan_mask
