"""Neighbor observation features: collection, ranking and packing."""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from flight_simulator import Aircraft

from config.settings import ObservationScaleSettings
from models.feature_transforms import (
    _bearing_sin_cos,
    _heading_rad,
    compute_ttca_dcpa,
    ego_components,
    map_positive_to_sym_bounded,
    map_sym,
    map_to_sym_unit,
    scalar_clamp,
)
from models.neighbor_ranking import NeighborRisk, neighbor_priority
from models.observation_settings import ObservationSettings
from outcomes.infringement import InfringementTuple
from simulator.route_section_store import is_past_section_end
from simulator.simulator import Simulator


@dataclass
class NeighborFeatures:
    """Container for neighbor features with normalization capabilities."""

    dx_fwd_nm: float  # neighbor forward offset in ego frame (nm)
    dy_right_nm: float  # neighbor right offset in ego frame (nm)
    dz_ft: float  # vertical offset (ft)
    dvx_fwd_nms: float  # relative forward velocity (nm/s)
    dvy_right_nms: float  # relative rightward velocity (nm/s)
    dvz_fpm: float  # relative vertical velocity (fpm)
    closing_nms: float  # line-of-sight closing rate (nm/s), negative = closing
    range_nm: float  # horizontal range (nm)
    bearing_sin: float  # sine of bearing to neighbor
    bearing_cos: float  # cosine of bearing to neighbor
    rel_hd_sin: float  # sine of relative projected heading
    rel_hd_cos: float  # cosine of relative projected heading
    ttca_s: float  # time to closest approach (s), computed from kinematics
    dcpa_nm: float  # distance at closest approach (nm)
    infringement_severity: float  # continuous [0,1]: 0=no infringement, 1=at conflict boundary
    head_on_closing_nms: float  # closing rate weighted by head-on factor (closing_nms × rel_hd_cos)

    def to_numpy(self, scales: ObservationScaleSettings) -> np.ndarray:
        """Convert to normalized numpy array for neural network input."""

        pos_xy_half_range_nm = float(abs(scales.rel_pos_xy_max))
        alt_half_range_ft = float(scales.rel_pos_z_range_ft)
        horiz_vel_range_kts = float(scales.rel_vel_range_kts)
        vert_vel_range_fpm = float(scales.rel_vz_range_fpm)
        ttca_clip_s = float(scales.ttca_max_s)
        dcpa_clip_nm = float(scales.dcpa_max_nm)

        return np.array(
            [
                map_sym(self.dx_fwd_nm, pos_xy_half_range_nm),
                map_sym(self.dy_right_nm, pos_xy_half_range_nm),
                map_sym(self.dz_ft, alt_half_range_ft),
                map_sym(self.dvx_fwd_nms * 3600.0, horiz_vel_range_kts),
                map_sym(self.dvy_right_nms * 3600.0, horiz_vel_range_kts),
                map_sym(self.dvz_fpm, vert_vel_range_fpm),
                map_sym(self.closing_nms * 3600.0, horiz_vel_range_kts),
                map_to_sym_unit(self.range_nm, 0.0, pos_xy_half_range_nm),
                float(self.bearing_sin),
                float(self.bearing_cos),
                float(self.rel_hd_sin),
                float(self.rel_hd_cos),
                map_to_sym_unit(self.ttca_s, 0.0, ttca_clip_s),
                map_positive_to_sym_bounded(self.dcpa_nm, dcpa_clip_nm, gain=3.0),
                float(self.infringement_severity),
                map_sym(self.head_on_closing_nms * 3600.0, horiz_vel_range_kts),
            ],
            dtype=np.float32,
        )


def _collect_neighbors(
    simulator: Simulator,
    ego_aircraft: Aircraft,
    all_current_infringements: dict[str, list[InfringementTuple]],
    settings: ObservationSettings,
    scales: ObservationScaleSettings,
):  # pylint: disable=too-many-locals,too-many-statements
    """Collect top-K neighbors around ego in ego-centric frame and normalize."""
    assert simulator.world is not None, "World is not initialized"

    epos = ego_aircraft.state.position_nm_nm_ft
    evel = ego_aircraft.state.velocity_knots
    ex, ey, ez = float(epos.x), float(epos.y), float(epos.z)
    evx, evy, evz_fpm = float(evel.x), float(evel.y), float(evel.z)

    ego_heading = _heading_rad(evx, evy)
    ego_hd_sin, ego_hd_cos = math.sin(ego_heading), math.cos(ego_heading)

    feats: list[NeighborFeatures] = []
    risks: list[NeighborRisk] = []
    for ac in sorted(simulator.world.aircraft, key=lambda a: a.call_sign):
        if (
            ac.call_sign == ego_aircraft.call_sign
            or not ac.under_control
            or ac.mission.completed
            or is_past_section_end(ac.route or [])
        ):
            continue

        pos = ac.state.position_nm_nm_ft
        vel = ac.state.velocity_knots
        dx = float(pos.x) - ex
        dy = float(pos.y) - ey
        dz_ft = float(pos.z) - ez
        dx_fwd, dy_right = ego_components(dx, dy, ego_heading)

        n_vx_kts = float(vel.x)
        n_vy_kts = float(vel.y)
        dvx_nms_glb = (n_vx_kts - evx) / 3600.0
        dvy_nms_glb = (n_vy_kts - evy) / 3600.0
        dvz_fpm = float(vel.z) - evz_fpm
        dvx_nms, dvy_nms = ego_components(dvx_nms_glb, dvy_nms_glb, ego_heading)

        raw_range_nm = float(math.hypot(dx_fwd, dy_right))
        range_nm = raw_range_nm + 1e-6
        b_sin, b_cos = _bearing_sin_cos(dy_right, dx_fwd)

        # Kinematic CPA metrics (purely from current relative position and velocity)
        t_cpa, d_cpa_nm = compute_ttca_dcpa(dx_fwd, dy_right, dvx_nms, dvy_nms, scales.ttca_max_s)
        t_cpa = scalar_clamp(t_cpa, 0.0, scales.ttca_max_s)

        # Relative heading from true simulator velocity vectors.
        nei_heading = _heading_rad(n_vx_kts, n_vy_kts)
        nei_sin, nei_cos = math.sin(nei_heading), math.cos(nei_heading)
        # sin(a-b) and cos(a-b)
        rel_hd_sin = nei_sin * ego_hd_cos - nei_cos * ego_hd_sin
        rel_hd_cos = nei_cos * ego_hd_cos + nei_sin * ego_hd_sin

        # Line-of-sight closing rate (negative = closing)
        closing_nms = 0.0
        if range_nm > 1e-6:
            closing_nms = (dx_fwd * dvx_nms + dy_right * dvy_nms) / range_nm

        # Head-on collision signal: closing rate weighted by head-on factor.
        # Large positive = dangerous head-on approach; negative = same-direction overtake.
        head_on_closing_nms = closing_nms * rel_hd_cos

        # Pairwise infringement severity between ego and this neighbor (0.0 = clear).
        neighbor_infrs = all_current_infringements.get(ac.call_sign, [])
        neighbor_severity = max(
            (t.value for t in neighbor_infrs if t.counterpart == ego_aircraft.call_sign),
            default=0.0,
        )

        speed_squared = dvx_nms * dvx_nms + dvy_nms * dvy_nms
        # Ranking retains slow finite approaches; the encoded CPA above
        # uses the stationary cutoff and bounded horizon.
        raw_cpa_s = (
            -(dx_fwd * dvx_nms + dy_right * dvy_nms) / speed_squared
            if speed_squared > 0.0
            else math.inf
        )
        raw_dcpa_nm = (
            math.hypot(dx_fwd + dvx_nms * raw_cpa_s, dy_right + dvy_nms * raw_cpa_s)
            if math.isfinite(raw_cpa_s) and raw_cpa_s > 0
            else raw_range_nm
        )
        risks.append(
            NeighborRisk(
                callsign=ac.call_sign,
                forecast_severity=neighbor_severity,
                severe_forecast=neighbor_severity >= scales.near_conflict_threshold,
                current_separation_threat=(
                    raw_range_nm <= scales.horizontal_safezone_radius_nm
                    and abs(dz_ft) <= scales.vertical_safezone_radius_ft
                ),
                raw_cpa_s=raw_cpa_s,
                cpa_horizontal_distance_nm=raw_dcpa_nm,
                vertical_relevance=abs(dz_ft) / scales.vertical_safezone_radius_ft,
                range_nm=raw_range_nm,
            )
        )

        feats.append(
            NeighborFeatures(
                dx_fwd_nm=dx_fwd,
                dy_right_nm=dy_right,
                dz_ft=dz_ft,
                dvx_fwd_nms=dvx_nms,
                dvy_right_nms=dvy_nms,
                dvz_fpm=dvz_fpm,
                closing_nms=closing_nms,
                range_nm=range_nm,
                bearing_sin=b_sin,
                bearing_cos=b_cos,
                rel_hd_sin=rel_hd_sin,
                rel_hd_cos=rel_hd_cos,
                ttca_s=t_cpa,
                dcpa_nm=d_cpa_nm,
                infringement_severity=neighbor_severity,
                head_on_closing_nms=head_on_closing_nms,
            )
        )

    feats = [
        feat
        for _, feat in sorted(
            zip(risks, feats, strict=True), key=lambda pair: neighbor_priority(pair[0])
        )
    ]
    feats = feats[: settings.max_neighbors]

    return _pack_and_normalize(feats, settings, scales)


def _pack_and_normalize(
    feats_list: Sequence[NeighborFeatures],
    settings: ObservationSettings,
    scales: ObservationScaleSettings,
):
    """Vectorize features, normalize/clamp to [-1,1], and pad to K with mask."""

    max_neighbours = settings.max_neighbors
    neighbor_feature_dim = settings.neighbor_dim

    nf = np.zeros((max_neighbours, neighbor_feature_dim), dtype=np.float32)
    mask = np.zeros((max_neighbours,), dtype=np.float32)
    for i, feat in enumerate(feats_list):
        nf[i] = feat.to_numpy(scales)
        mask[i] = 1.0

    return nf, mask
