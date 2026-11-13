"""Functionality for creating tensor observations from the simulator."""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from actions import action_mask
from actions.rewards import get_time_deviations
from config.config import Config
from flight_simulator import Aircraft, Vec3, World
from simulator.simulator import Simulator
from utils.infringement_utils import get_world_infringements


@dataclass
class AircraftScalarFeatures:
    """Container for aircraft-level scalar features with normalization capabilities."""

    pos_x: float  # Global x position (nm)
    pos_y: float  # Global y position (nm)
    pos_z: float  # Global z position (ft)
    vel_xy: float  # Velocity on xy plane (kts)
    vz: float  # Vertical velocity (fpm)
    heading_sin: float  # Sine of current heading
    heading_cos: float  # Cosine of current heading
    time_to_target: float  # Time to target (s)
    time_deviation: float  # Schedule deviation (s)
    predicted_infringement: float  # Predicted conflict risk [0,1]
    time_to_conflict: float  # Urgency of nearest predicted conflict [0,1]: 1=imminent, 0=none/far
    under_control: float  # Under ATC control (0/1)

    def to_numpy(self) -> np.ndarray:
        """Convert to normalized numpy array for neural network input."""

        return np.array(
            [
                map_sym(self.pos_x, Config.GLOBAL_POS_XY_MAX),
                map_sym(self.pos_y, Config.GLOBAL_POS_XY_MAX),
                map_sym(self.pos_z, Config.GLOBAL_POS_Z_MAX),
                map_to_sym_unit(self.vel_xy, 0.0, Config.VEL_XY_MAX),
                map_sym(self.vz, Config.VEL_Z_MAX),
                float(self.heading_sin),
                float(self.heading_cos),
                map_time_to_target(self.time_to_target),
                map_deviation(self.time_deviation),
                map_to_sym_unit(
                    self.predicted_infringement,
                    Config.PREDICTED_INFRINGEMENT_MIN,
                    Config.PREDICTED_INFRINGEMENT_MAX,
                ),
                _map01_to_sym(self.time_to_conflict),
                float(self.under_control),
            ],
            dtype=np.float32,
        )


@dataclass
class GlobalFeatures:
    """Global context features with normalization."""

    time_s: float  # world age in seconds
    active_aircraft_count: float  # number of under-control, active aircraft
    mean_time_deviation_s: float  # mean absolute schedule deviation across active aircraft
    max_predicted_infringement: float  # in [0, 1]

    def to_numpy(self) -> np.ndarray:
        """Convert to normalized numpy array for neural network input."""

        horizon = float(Config.EPISODE_STEPS * Config.TIME_BETWEEN_ACTIONS)
        t_norm = map_to_sym_unit(self.time_s, 0.0, horizon)
        active_norm = map_to_sym_unit(self.active_aircraft_count, 0.0, float(Config.N_AIRCRAFT_MAX))
        # mean_time_deviation_s is a mean of ABSOLUTE deviations, so it is non-negative;
        # map the log magnitude onto the full [-1,1] range rather than only its positive half.
        dev_norm = _map01_to_sym(
            map_signed_log(
                max(0.0, self.mean_time_deviation_s),
                float(Config.TIME_DEVIATION_MAX),
                float(Config.TIME_DEVIATION_LOG_FLOOR_S),
            )
            if bool(getattr(Config, "USE_LOG_DEVIATION_OBS", False))
            else _normalize(self.mean_time_deviation_s, 0.0, float(Config.TIME_DEVIATION_MAX))
        )
        infr_norm = map_to_sym_unit(
            self.max_predicted_infringement,
            float(Config.PREDICTED_INFRINGEMENT_MIN),
            float(Config.PREDICTED_INFRINGEMENT_MAX),
        )
        global_array = np.array([t_norm, active_norm, dev_norm, infr_norm], dtype=np.float32)

        assert (
            len(global_array) == Config.GLOBAL_DIM
        ), f"Global feature dimension mismatch: {len(global_array)} != {Config.GLOBAL_DIM}"

        return global_array


@dataclass
class WaypointRelFeatures:
    """Relative waypoint features with normalization capabilities."""

    # Global coordinates of the waypoint
    pos_x: float  # waypoint global x (nm)
    pos_y: float  # waypoint global y (nm)
    pos_z: float  # waypoint global z (ft)
    dx: float  # x offset to next waypoint (nm)
    dy: float  # y offset to next waypoint (nm)
    dz: float  # altitude offset to next waypoint (ft)
    dist: float  # horizontal distance to next waypoint (nm)
    bearing_sin: float  # sine of bearing to next waypoint
    bearing_cos: float  # cosine of bearing to next waypoint
    vel_target: float  # target velocity at next waypoint (kts)
    vel_target_available: float  # is target velocity available (0/1)

    def to_numpy(self) -> np.ndarray:
        """Convert to normalized numpy array for neural network input."""

        pos_xy_half_range = float(abs(Config.GLOBAL_POS_XY_MAX))
        dz_half_range = float(Config.REL_POS_Z_RANGE_FT)
        dist_clip_nm = float(abs(Config.GLOBAL_POS_XY_MAX))

        return np.array(
            [
                # Global waypoint position
                map_sym(self.pos_x, Config.GLOBAL_POS_XY_MAX),
                map_sym(self.pos_y, Config.GLOBAL_POS_XY_MAX),
                map_sym(self.pos_z, Config.GLOBAL_POS_Z_MAX),
                map_sym(self.dx, pos_xy_half_range),
                map_sym(self.dy, pos_xy_half_range),
                map_sym(self.dz, dz_half_range),
                map_to_sym_unit(self.dist, 0.0, dist_clip_nm),
                self.bearing_sin,
                self.bearing_cos,
                (
                    map_to_sym_unit(self.vel_target, 0.0, Config.VEL_XY_MAX)
                    if self.vel_target_available > 0
                    else 0.0
                ),
                self.vel_target_available,
            ],
            dtype=np.float32,
        )

@dataclass
class AircraftActionFeatures:
    """Encodes recent action history for an aircraft."""

    clearance_id: int
    valid: float

    def to_numpy(self) -> np.ndarray:
        """Convert to one-hot action-history feature vector.

        Shape:
            [1 + Config.N_CLEARANCES]

        Layout:
            [valid_flag, one_hot_clearance_0, ..., one_hot_clearance_N]
        """
        out = np.zeros(
            1 + int(Config.N_CLEARANCES),
            dtype=np.float32,
        )

        out[0] = float(self.valid)

        if self.valid > 0.0 and 0 <= self.clearance_id < int(Config.N_CLEARANCES):
            out[1 + self.clearance_id] = 1.0

        return out


# These four helpers are the hottest code in the observation pipeline: profiling
# build_global_observation showed 1,192 np.clip calls PER BUILD, accounting for 42% of its
# runtime. np.clip on a Python scalar is pathologically slow — it dispatches through
# _wrapfunc -> _wrapit -> ndarray.clip -> _clip, allocating on the way — while min/max on a
# float is a couple of comparisons. The arithmetic is identical for scalars.
#
# NaN note: np.clip propagates NaN, and so does this — both comparisons are False for NaN,
# so it falls through and returns the value unchanged. Behaviour is preserved. (The
# observation audit reports zero non-finite values, so this is theoretical either way.)
def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _normalize(value: float, vmin: float, vmax: float) -> float:
    """Min-max normalization with clamping to [0, 1]."""
    if vmax == vmin:
        return 0.5
    return _clamp((value - vmin) / (vmax - vmin), 0.0, 1.0)


def _normalize_sym(value: float, scale: float) -> float:
    """Map a value to [-1, 1] using symmetric scale with clamping."""
    if scale <= 0:
        return 0.0
    return _clamp(value / scale, -1.0, 1.0)


def _map01_to_sym(x01: float) -> float:
    """Map [0,1] -> [-1,1]."""
    return 2.0 * _clamp(x01, 0.0, 1.0) - 1.0


def map_sym(value: float, half_range: float) -> float:
    """Helper: symmetric normalization to [-1, 1] with clamping."""
    return _normalize_sym(value, half_range)


def map_to_sym_unit(value: float, vmin: float, vmax: float) -> float:
    """Helper: min-max normalize to [0,1], then map to [-1,1]."""
    return _map01_to_sym(_normalize(value, vmin, vmax))


def map_signed_log(value: float, vmax: float, floor: float) -> float:
    """Sign-preserving log-magnitude normalisation to [-1, 1].

        out = sign(v) * ln(1 + |v| / floor) / ln(1 + vmax / floor)

    Used for schedule deviation, where the quantity spans three orders of magnitude but
    every decision boundary that matters sits at the bottom of the range. A linear map to
    +/-900 s encodes a 10 s deviation as 0.011 -- below the noise floor of the downstream
    LayerNorm -- and squeezes the whole +/-60..120 s tier ladder into 13% of the range.
    The log spends resolution where the tiers are and takes it from the 300 s+ tail, where
    the exact value does not change what the agent should do.

    ``floor`` is the resolution knee: magnitudes below it are treated as roughly equivalent,
    which keeps the encoding from amplifying sub-simulator-resolution jitter around zero.
    Sign is preserved because early and late are different problems with different fixes.
    """
    if vmax <= 0.0 or floor <= 0.0:
        return 0.0
    denom = math.log1p(vmax / floor)
    if denom <= 0.0:
        return 0.0
    magnitude = math.log1p(min(abs(float(value)), vmax) / floor) / denom
    return _clamp(math.copysign(magnitude, value), -1.0, 1.0)


def map_deviation(value: float) -> float:
    """Encode a signed schedule deviation (s) for the observation.

    ``Config.USE_LOG_DEVIATION_OBS`` selects logarithmic or linear encoding.
    """
    if bool(getattr(Config, "USE_LOG_DEVIATION_OBS", False)):
        return map_signed_log(
            value,
            float(Config.TIME_DEVIATION_MAX),
            float(Config.TIME_DEVIATION_LOG_FLOOR_S),
        )
    return map_sym(value, float(Config.TIME_DEVIATION_MAX))


def map_time_to_target(value: float) -> float:
    """Encode a non-negative time-to-target (s) to [-1, 1].

    Log-scaled for the same reason as deviation: the difference between 60 s and 120 s out
    changes what the agent should do, while the difference between 2000 s and 2500 s does not.
    """
    if bool(getattr(Config, "USE_LOG_DEVIATION_OBS", False)):
        return _map01_to_sym(
            map_signed_log(
                max(0.0, float(value)),
                float(Config.TIME_TO_TARGET_MAX),
                float(Config.TIME_TO_TARGET_LOG_FLOOR_S),
            )
        )
    return map_to_sym_unit(
        value, Config.TIME_TO_TARGET_MIN, Config.TIME_TO_TARGET_MAX
    )




def _heading_sin_cos(vx: float, vy: float) -> Tuple[float, float]:
    """Return sin/cos of heading based on horizontal velocity vector."""
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0, 1.0
    angle = math.atan2(vx, vy)
    return math.sin(angle), math.cos(angle)


def _bearing_sin_cos(dx: float, dy: float) -> Tuple[float, float]:
    """Return sin/cos of bearing from current pos to target (dx, dy)."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0, 1.0
    angle = math.atan2(dx, dy)
    return math.sin(angle), math.cos(angle)


def _distance(dx: float, dy: float) -> float:
    return float(math.hypot(dx, dy))


def _compute_aircraft_scalar(
    pos: Vec3,
    vel: Vec3,
    ttt: float,
    td: float,
    pin: float,
    tcf: float,
    under_control: bool,
) -> AircraftScalarFeatures:
    """Compute aircraft-level scalar features as a readable datamodel."""
    vel_xy = float(math.hypot(float(vel.x), float(vel.y)))
    h_sin, h_cos = _heading_sin_cos(float(vel.x), float(vel.y))

    vz = float(vel.z)

    return AircraftScalarFeatures(
        pos_x=pos.x,
        pos_y=pos.y,
        pos_z=pos.z,
        vel_xy=vel_xy,
        vz=vz,
        heading_sin=h_sin,
        heading_cos=h_cos,
        time_to_target=ttt,
        time_deviation=td,
        predicted_infringement=pin,
        time_to_conflict=tcf,
        under_control=float(under_control),
    )


def _collect_action_history_features(
    call_sign: str,
    global_action_history: List[Dict[str, Any]],
) -> Tuple[List[AircraftActionFeatures], np.ndarray]:
    """Collect recent action history features for one aircraft.

    Returns:
        actions: list of AircraftActionFeatures
        actions_np: array with shape [Config.MAX_HISTORY, 1 + Config.N_CLEARANCES]
    """
    actions: List[AircraftActionFeatures] = []

    for entry in reversed(global_action_history):
        if entry.get("call_sign") != call_sign:
            continue

        raw_action = int(entry.get("action", 0))

        # If entry["action"] is already a clearance id, this is fine.
        # If it is a flattened global action, decode it first.
        if raw_action >= int(Config.N_CLEARANCES):
            from network.rlm import decode_action

            _, clearance_id = decode_action(raw_action)
        else:
            clearance_id = raw_action

        actions.append(
            AircraftActionFeatures(
                clearance_id=int(clearance_id),
                valid=1.0,
            )
        )

        if len(actions) >= int(Config.MAX_HISTORY):
            break

    # Pad with invalid history entries
    while len(actions) < int(Config.MAX_HISTORY):
        actions.append(
            AircraftActionFeatures(
                clearance_id=0,
                valid=0.0,
            )
        )

    actions_np = np.stack(
        [a.to_numpy() for a in actions],
        axis=0,
    ).astype(np.float32)

    return actions, actions_np


_WP_CACHE: Dict[str, Any] = {"key": None, "xy": {}}


def _translated_waypoints(simulator: Simulator, T_obs) -> Dict[str, Tuple[float, float]]:
    """name -> translated (x, y) for every waypoint of the current scenario, via translate_point.

    Cached per (frame, scenario): T_obs is drawn once per episode and the waypoint set belongs
    to the scenario (simulator.initial_world, replaced on every reset), so the key changes
    exactly when the translated positions can.
    """
    key = (np.asarray(T_obs).tobytes(), id(simulator.initial_world))
    if _WP_CACHE["key"] != key:
        xy = {}
        for name, wp in simulator.world.waypoints.items():
            xy[name] = translate_point(wp.position_nm.x, wp.position_nm.y, T_obs)
        _WP_CACHE["key"] = key
        _WP_CACHE["xy"] = xy
    return _WP_CACHE["xy"]


def _collect_waypoints_and_next_wp_rel(
    aircraft: Aircraft,
    simulator: Simulator,
    current_pos: Any,  # flight_simulator position object with .x, .y, .z attributes
    T_obs
) -> Tuple[List[Tuple[float, float, float]], WaypointRelFeatures, List[WaypointRelFeatures]]:
    """Collect global waypoints and relative features for the flight plan."""

    assert simulator.world is not None, "World is not initialized"

    waypoints_glb: List[Tuple[float, float, float]] = []
    rel_waypoints: List[WaypointRelFeatures] = []

    if aircraft.route:
        # Translated waypoint positions are fixed for the episode (static waypoints, one frame
        # T_obs per episode), and the aircraft's own translated position is fixed for this call,
        # so both are computed once instead of once per route point. Same function, same inputs:
        # the observation is byte-identical. The per-point `world.waypoints` access was the
        # hot spot -- it crosses into Rust every time.
        wp_xy = _translated_waypoints(simulator, T_obs)
        cx, cy = translate_point(current_pos.x, current_pos.y, T_obs)
        current_pos_obs = Vec3(cx, cy, current_pos.z)
        for rp in aircraft.route[: Config.MAX_WAYPOINTS]:
            txy = wp_xy.get(rp.waypoint_name)
            if txy is None:
                continue
            alt = ((rp.min_altitude_ft or 0) + (rp.max_altitude_ft or 0)) / 2 or float(
                current_pos.z
            )
            glb_wp = (
                float(txy[0]),
                float(txy[1]),
                float(alt),
            )

            waypoints_glb.append(glb_wp)
            rel_waypoints.append(
                _wp_rel_from_glb(glb_wp, current_pos_obs, rp.target_speed)
            )
    if rel_waypoints:
        next_wp_rel = rel_waypoints[0]
    else:
        next_wp_rel = WaypointRelFeatures(
            pos_x=0.0,
            pos_y=0.0,
            pos_z=0.0,
            dx=0.0,
            dy=0.0,
            dz=0.0,
            dist=0.0,
            bearing_sin=0.0,
            bearing_cos=1.0,
            vel_target=0.0,
            vel_target_available=0.0,
        )

    return waypoints_glb, next_wp_rel, rel_waypoints


def _wp_rel_from_glb(
    glb_wp: Tuple[float, float, float],
    current_pos: Any,  # flight_simulator position object with .x, .y, .z attributes
    vel_target: Optional[float],
) -> WaypointRelFeatures:
    pos_x, pos_y, pos_z = glb_wp
    dx = pos_x - float(current_pos.x)
    dy = pos_y - float(current_pos.y)
    dz = pos_z - float(current_pos.z)
    dist = _distance(dx, dy)
    b_sin, b_cos = _bearing_sin_cos(dx, dy)
    vel_target = float(vel_target) if vel_target is not None else 0.0
    return WaypointRelFeatures(
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=pos_z,
        dx=dx,
        dy=dy,
        dz=dz,
        dist=dist,
        bearing_sin=b_sin,
        bearing_cos=b_cos,
        vel_target=vel_target,
        vel_target_available=1.0 if vel_target > 0.0 else 0.0,
    )

def translate_point(x: float, y: float, T: np.ndarray) -> Tuple[float, float]:
    """Apply homogeneous translation matrix."""
    p = np.array([x, y, 1.0], dtype=np.float32)
    p_t = T @ p
    return float(p_t[0]), float(p_t[1])

def get_valid_aircraft_list(world: World) -> List[Aircraft]:
    aircraft = [ac for ac in world.aircraft if ac.under_control and not ac.mission.completed]
    if len(aircraft) == 0:
        # print("NO AIRCRAFT UNDER CONTROL")
        aircraft = [ac for ac in world.aircraft if not ac.mission.completed]
    return aircraft#ac.under_control and


def build_global_observation(
        simulator: Simulator,
        curr_world: Optional[World],
        # prev_world: Optional[World],
        global_action_history,
        perm,
        route_trombone_store,
        T_obs,
        aircraft_list: Optional[List[Aircraft]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
    """Build global observation for single-agent AR policy.

    ``aircraft_list`` (optional) is the exact, ordered set of aircraft to place in the slots.
    None => every active aircraft in world order (get_valid_aircraft_list), the behaviour of
    every run so far. The windowed env passes its landing-order window here.
    """
    # GPT: We need to add an episodically relative position of the airport to every observation

    assert simulator.world is not None, "World is not initialized"
    assert curr_world is not None, "Prev world is not initialized"
    # assert prev_world is not None, "Prev world is not initialized"

    # Look-ahead window using simulator time. Use the full rollout horizon so the
    # observation sees every predicted conflict that the conflict reward considers.
    now_s = float(simulator.world.age_in_seconds)
    horizon_s = Config.EPISODE_STEPS * Config.TIME_BETWEEN_ACTIONS

    # infr_map = get_world_infringements(prev_world, now_s, horizon_s)
    # td_map, _ = get_time_deviations(prev_world)
    infr_map = get_world_infringements(curr_world, now_s, horizon_s)
    # Simulator-obtained per-aircraft schedule deviation (predicted landing vs AMAN) from the rollout.
    td_map, _ = get_time_deviations(curr_world)

    # Stable list of active aircraft under control
    if aircraft_list is None:
        aircraft_list = get_valid_aircraft_list(simulator.world)

    n_air_max = int(Config.N_AIRCRAFT_MAX)
    feat_dim = int(Config.AIRCRAFT_SCALAR_LEN)
    n_actions = int(Config.N_CLEARANCES)

    # Global features (raw values)
    time_s = float(simulator.world.age_in_seconds)
    active_cnt = float(len(aircraft_list))

    dev_vals = [abs(float(td_map.get(ac.call_sign, 0.0))) for ac in aircraft_list]

    dev_mean = float(np.mean(dev_vals)) if dev_vals else 0.0
    # `proximity`, not `value`: the observation deliberately reads the WIDE detection band
    # (non-zero out to 10 NM) while the reward reads the narrow severity band (0 beyond
    # 5 NM). The agent must be able to watch a conflict develop long before it is charged
    # for it.
    max_infr = max(
        (float(t.proximity) for lst in infr_map.values() for t in lst), default=0.0
    )

    gf = GlobalFeatures(
        time_s=time_s,
        active_aircraft_count=active_cnt,
        mean_time_deviation_s=dev_mean,
        max_predicted_infringement=max_infr,
    )

    # Per-aircraft features
    ac_mat = np.zeros((n_air_max, feat_dim), dtype=np.float32)

    # Masks
    mask_aircraft = np.zeros((n_air_max,), dtype=np.int8)
    mask_action_per_ac = np.zeros((n_air_max, n_actions), dtype=np.int8)

    # Flight plans: per-aircraft relative waypoint sequences + mask
    max_wps = int(Config.MAX_WAYPOINTS)
    wp_dim = int(Config.FLIGHT_PLAN_WAYPOINT_DIM)
    flight_plans_rel = np.zeros((n_air_max, max_wps, wp_dim), dtype=np.float32)
    flight_plan_mask = np.zeros((n_air_max, max_wps), dtype=np.int8)

    # Per-aircraft recent action history
    # Shape: [N_AIRCRAFT_MAX, MAX_HISTORY, 1 + N_CLEARANCES]
    # Feature layout per history step:
    # [valid_flag, one_hot_clearance_0, ..., one_hot_clearance_N]
    action_hist_dim = 1 + int(Config.N_CLEARANCES)

    action_histories = np.zeros(
        (
            n_air_max,
            int(Config.MAX_HISTORY),
            action_hist_dim,
        ),
        dtype=np.float32,
    )

    # Slot->callsign mapping for env
    aircraft_order: List[str] = [""] * n_air_max

    for i, ac in enumerate(aircraft_list[:n_air_max]):
        slot = int(perm[i])
        vel = ac.state.velocity_knots

        pos = ac.state.position_nm_nm_ft

        tx, ty = translate_point(pos.x, pos.y, T_obs)

        pos_obs = Vec3(
            tx,
            ty,
            pos.z
        )

        # Scalar features
        td = float(td_map.get(ac.call_sign, 0.0))

        # Predicted infringement proxy: max value within look-ahead
        seq = infr_map.get(ac.call_sign, [])
        pin = max((float(t.proximity) for t in seq), default=0.0)

        # Time-to-conflict urgency of the nearest predicted conflict (1=imminent, 0=none/far).
        # seq is sorted by timestamp ascending, so seq[0] is the earliest predicted conflict.
        if seq:
            ttc = float(seq[0].timestamp) - now_s
            tcf = max(0.0, 1.0 - ttc / float(Config.NEAR_CONFLICT_TIME_S))
        else:
            tcf = 0.0

        # Time to target
        # ttt = float(simulator.world.calculate_time_to_target_arrival(ac.call_sign))
        ttt = float(curr_world.calculate_time_to_target_arrival(ac.call_sign))

        # Only scalar features go into per-aircraft vector now
        ac_scalar = _compute_aircraft_scalar(
            pos=pos_obs, vel=vel, ttt=ttt, td=td, pin=pin, tcf=tcf, under_control=ac.under_control
        )
        vec = ac_scalar.to_numpy()
        ac_mat[slot, : len(vec)] = vec.astype(np.float32)
        mask_aircraft[slot] = np.int8(1)

        # Build full flight plan relative features up to MAX_WAYPOINTS
        waypoints_global, next_wp_rel, rel_waypoints = _collect_waypoints_and_next_wp_rel(
            aircraft=ac, simulator=simulator, current_pos=pos, T_obs=T_obs
        )
        for j, rel_wp in enumerate(rel_waypoints[:max_wps]):
            flight_plans_rel[slot, j] = rel_wp.to_numpy()
            flight_plan_mask[slot, j] = np.int8(1)

        _, act_hist_np = _collect_action_history_features(
            call_sign=ac.call_sign,
            global_action_history=global_action_history,
        )
        action_histories[slot] = act_hist_np

        # Per-aircraft action mask
        next_wp_vel_target = (
            next_wp_rel.vel_target if next_wp_rel.vel_target_available > 0 else None
        )
        route_names = [rp.waypoint_name for rp in (ac.route or [])]
        per_mask = action_mask.compute_action_mask(
            ac_name=ac.call_sign,
            # waypoints_global are in the translated frame (T_obs), so the position must be too
            # (Config.MASK_FRAME_FIX selects the frame convention)
            current_pos=pos_obs if Config.MASK_FRAME_FIX else pos,
            target_vel=ac.target_state.speed or vel.length(),
            current_vel=vel,
            wps_list=waypoints_global,
            route_name=ac.route_name or "UNKNOWN",
            route=ac.route,
            design_data=simulator.parsed_designs[ac.aircraft_type],
            wp_vel_target=next_wp_vel_target,
            route_waypoint_names=route_names,
            target_heading_deg=ac.target_state.heading,
            predicted_infringement=pin,
            route_trombone_store=route_trombone_store,
        )[: Config.N_CLEARANCES].astype(np.int8)
        mask_action_per_ac[slot, : len(per_mask)] = per_mask

        aircraft_order[slot] = ac.call_sign
        # assert aircraft_order is not None and len(aircraft_order) > 0
        # if aircraft_order is not None and len(aircraft_order) > 0:
        #     print(aircraft_order)
    return {
        "global": gf.to_numpy(),
        "aircraft": ac_mat,
        "mask_aircraft": mask_aircraft,
        "mask_action_per_ac": mask_action_per_ac,
        "flight_plans_rel": flight_plans_rel,
        "flight_plan_mask": flight_plan_mask,
        "action_histories": action_histories,
    }, aircraft_order
