"""Valid-clearance masks for the point-merge set v2pms (see actions_v2pms.py).

v2's speed, turn and skip masks, plus DIRECT_TO_MERGE_POINT while the flight is on its
sequencing leg. Lengthen/shorten trombone are never legal.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from config.config import Config
from flight_simulator import Vec3, RoutePoint
from simulator.design_parser import AircraftDesignData
from simulator.route_section_store import RouteSectionStore

from . import action_mask_v2 as _v2
from .actions_v2pms import Action


def compute_action_mask(
    ac_name: str,
    current_pos: Vec3,
    target_vel: float,
    current_vel: Vec3,
    wps_list: List[Tuple[float, float, float]],
    route_name: str,
    route: list[RoutePoint],
    design_data: AircraftDesignData,
    *,
    wp_vel_target: Optional[float] = None,
    route_waypoint_names: Optional[Sequence[str]] = None,
    target_heading_deg: Optional[float] = None,
    predicted_infringement: Optional[float] = None,
    route_trombone_store: RouteSectionStore,
) -> np.ndarray:
    """Return boolean mask for valid actions (False / zero = invalid action)"""
    mask = np.zeros(Config.NUM_ACTIONS, dtype=bool)
    mask[int(Action.DO_NOTHING)] = True

    mask = _v2._mask_speed_actions(mask, target_vel, design_data, wp_vel_target)

    if len(wps_list) > Config.WAYPOINTS_TO_LEAVE_AT_MINIMUM:
        mask = _v2._mask_vectoring_actions(mask, current_pos, current_vel, target_heading_deg, wps_list)
        mask = _v2._mask_skip_waypoints_actions(mask, current_pos, wps_list)
        if route_trombone_store.can_direct_to_merge(ac_name, route):
            mask[int(Action.DIRECT_TO_MERGE_POINT)] = True

    mask = _v2._mask_risk_aware_no_op_action(mask=mask, predicted_infringement=predicted_infringement)

    if Config.CUSTOM_ACTION_MASK is not None:
        for i, val in enumerate(Config.CUSTOM_ACTION_MASK):
            if i < len(mask) and val > 0:
                mask[i] = True

    assert mask.any(), "action_mask ended up empty: this must never happen"
    return mask
