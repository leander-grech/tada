"""Functionality for creating tensor observations from the simulator."""

from typing import AbstractSet, Any, Sequence

import numpy as np
from flight_simulator import Aircraft

from actions import action_mask
from actions.action_table import NUM_ACTIONS
from config.observation_fields import GLOBAL_CONTEXT_FIELDS
from config.settings import ActionSettings, ObservationScaleSettings
from models.feature_transforms import (
    signed_time_remaining,
)
from models.neighbors import _collect_neighbors
from models.observation_features import (
    _collect_waypoints_and_rel_features,
    _compute_aircraft_scalar,
    _compute_flight_plans_rel,
)
from models.observation_settings import ObservationSettings
from outcomes.infringement import InfringementTuple
from simulator.route_section_store import RouteSectionStore

# Bump on any change that alters outputs; checkpoints record this token and reject mismatches.
OBS_VERSION = "ma-observation-v4"


def make_agent_observations_from_simulator(
    simulator: Any,
    agents_to_observe: list[str],
    current_infringements: dict[str, list[InfringementTuple]],
    deviation_proxies: dict[str, float],
    route_trombone_store: RouteSectionStore,
    global_context: Sequence[float] | None = None,
    last_non_zero_action_times: dict[str, float] | None = None,
    allowed_to_act: AbstractSet[str] | None = None,
    locked_out_agents: AbstractSet[str] | None = None,
    aircraft_airports: dict[str, str | None] | None = None,
    *,
    settings: ObservationSettings,
    scales: ObservationScaleSettings,
    action_settings: ActionSettings,
) -> dict[str, Any]:
    """Create observations for specific agents only."""
    agent_observations = {}
    allowed = allowed_to_act
    locked = locked_out_agents or frozenset()
    airports = aircraft_airports or {}

    for agent_id in agents_to_observe:
        ac = simulator.world.get_aircraft_by_callsign(agent_id)
        observation = _generate_single_aircraft_observation(
            ac,
            simulator,
            deviation_proxies,
            current_infringements,
            route_trombone_store,
            global_context=global_context,
            last_non_zero_action_times=last_non_zero_action_times,
            locked_out=agent_id in locked,
            airport=airports.get(agent_id),
            settings=settings,
            scales=scales,
            action_settings=action_settings,
        )
        if allowed is not None and agent_id not in allowed:
            forced_noop_mask = np.zeros((NUM_ACTIONS,), dtype=np.uint8)
            forced_noop_mask[int(action_mask.Action.DO_NOTHING)] = 1
            observation["action_mask"] = forced_noop_mask
        agent_observations[agent_id] = observation

    return agent_observations


def _fixed_width_context_vector(
    raw_values: Sequence[float] | None,
    width: int,
) -> np.ndarray:
    """Copy an optional context vector into a fixed-width [0,1] array."""
    context = np.zeros((width,), dtype=np.float32)
    if raw_values is None:
        return context

    raw = np.asarray(raw_values, dtype=np.float32)
    used_dim = min(int(raw.shape[0]), int(width))
    if used_dim > 0:
        context[:used_dim] = raw[:used_dim]
    np.clip(context, 0.0, 1.0, out=context)
    return context


def _global_context_vector(global_context: Sequence[float] | None) -> np.ndarray:
    """Return fixed-width shared global context."""
    return _fixed_width_context_vector(global_context, len(GLOBAL_CONTEXT_FIELDS))


def _generate_single_aircraft_observation(  # pylint: disable=too-many-locals,too-many-statements
    aircraft: Aircraft,
    simulator: Any,
    deviation_proxies: dict[str, float],
    current_infringements: dict[str, list[InfringementTuple]],
    route_trombone_store: RouteSectionStore,
    global_context: Sequence[float] | None = None,
    last_non_zero_action_times: dict[str, float] | None = None,
    *,
    locked_out: bool = False,
    airport: str | None = None,
    settings: ObservationSettings,
    scales: ObservationScaleSettings,
    action_settings: ActionSettings,
):
    target_vel = aircraft.target_state.speed
    current_pos, current_vel = (
        aircraft.state.position_nm_nm_ft,
        aircraft.state.velocity_knots,
    )
    target_heading = aircraft.target_state.heading

    deviation_proxy = float(deviation_proxies.get(aircraft.call_sign, 0.0))
    current_time_s = float(simulator.world.age_in_seconds)
    target = next(
        (
            target
            for target in simulator.world.arrival_targets
            if target.ac_callsign == aircraft.call_sign
        ),
        None,
    )
    if target is None:
        raise ValueError(f"arrival target missing for aircraft {aircraft.call_sign}")
    ttt = signed_time_remaining(target.eta_s, current_time_s)

    waypoints_global, rel_waypoints = _collect_waypoints_and_rel_features(
        aircraft=aircraft, simulator=simulator, current_pos=current_pos, settings=settings
    )

    flight_plans_rel, flight_plan_mask = _compute_flight_plans_rel(
        rel_waypoints=rel_waypoints, settings=settings, scales=scales
    )

    neighbor_features, neighbor_mask = _collect_neighbors(
        simulator,
        aircraft,
        all_current_infringements=current_infringements,
        settings=settings,
        scales=scales,
    )
    critic_global_context = _global_context_vector(global_context)

    last_action_s = current_time_s
    if last_non_zero_action_times is not None:
        last_action_s = float(last_non_zero_action_times.get(aircraft.call_sign, current_time_s))
    time_since_last_action_s = max(0.0, current_time_s - last_action_s)

    aircraft_scalar = _compute_aircraft_scalar(
        vel=current_vel,
        ttt=ttt,
        deviation_proxy=deviation_proxy,
        under_control=aircraft.under_control and not locked_out,
        time_since_last_action=time_since_last_action_s,
        timing_available=aircraft.call_sign in deviation_proxies,
    )

    ac_design = simulator.parsed_designs[aircraft.aircraft_type]

    mask = action_mask.compute_action_mask(
        ac_name=aircraft.call_sign,
        current_pos=current_pos,
        target_vel=target_vel or current_vel.length(),
        current_vel=current_vel,
        wps_list=waypoints_global,
        route=aircraft.route,
        design_data=ac_design,
        target_heading_deg=target_heading,
        route_trombone_store=route_trombone_store,
        ego_infringements=current_infringements.get(aircraft.call_sign),
        airport=airport,
        settings=action_settings,
    )[:NUM_ACTIONS].astype(np.uint8)

    return {
        "observations": {
            "aircraft_scalar": aircraft_scalar.to_numpy(settings, scales),
            "global_context": critic_global_context,
            "flight_plans_rel": flight_plans_rel,
            "flight_plan_mask": flight_plan_mask,
            "neighbor_features": neighbor_features,
            "neighbor_mask": neighbor_mask,
        },
        "action_mask": mask,
    }
