"""Module for the simulator"""

import os
import random
from typing import Any, Mapping, cast

from flight_simulator import (
    Aircraft,
    AircraftCommand,
    AircraftDesign,
    DataRecorder,
    Mission,
    MissionDifficulty,
    RoutePoint,
    ScenarioConfig,
    World,
    generate_random_mission,
    load_config,
    load_world,
    set_seed,
    simulate_n_next_states,
    simulate_next_state,
)

from common.runtime_log import runtime_print
from common.world_digest import world_digest
from config.settings import SimulatorSettings
from config.simulation_config_path import SimulationConfigPath
from geometry.vectoring import convex_hull, resolve_waypoint_position, shrink_polygon
from simulator.commands import TimedCommandList
from simulator.design_parser import (
    AircraftDesignData,
    parse_all_aircraft_designs,
)
from simulator.outcome_port import ScenarioOutcomeOwner
from simulator.route_section_store import RouteSectionStore

SIMULATOR_TIMESTEP_MS = 1000


class Simulator:
    """Simulator class for the environment"""

    def __init__(
        self,
        difficulty: MissionDifficulty,
        scenario_config_path: SimulationConfigPath,
        route_trombone_store: RouteSectionStore,
        *,
        settings: SimulatorSettings,
        outcome_owner: ScenarioOutcomeOwner | None = None,
    ):
        """Initialize the simulator with config and load initial state."""
        self.outcome_owner = outcome_owner
        self._prepared_world_token = None
        self.settings = settings
        self.scenario_config: ScenarioConfig | None = None
        self.recorder = DataRecorder()
        self.world: World | None = None
        self.initial_world: World | None = None
        self.designs: Mapping[str, AircraftDesign] = {}
        self.parsed_designs: dict[str, AircraftDesignData] = {}
        self.seed_set = False
        self.last_scenario_generation_attempts = 0
        self.capture_generation = False
        self.raw_generated_world_json = ""
        self.route_trombone_store = route_trombone_store
        self._load_configs(scenario_config_path, difficulty)

    def advance_multiple_steps(
        self, steps: int, world: World, time_between_actions: int | None = None
    ) -> World:
        """Advance the simulator forward by 'steps' timesteps."""
        if time_between_actions is None:
            time_between_actions = self.settings.time_between_actions
        return simulate_n_next_states(
            steps * time_between_actions,
            SIMULATOR_TIMESTEP_MS,
            world,
            [],
        )

    def apply_action_and_advance(
        self,
        actions: TimedCommandList,
        time_between_actions: int | None = None,
    ):
        """
        Apply the given actions, then advance the simulator by TIME_BETWEEN_ACTIONS timesteps.
        """

        if time_between_actions is None:
            time_between_actions = self.settings.time_between_actions
        valid_actions = TimedCommandList(
            commands=[action for action in actions.commands if action.command is not None]
        )

        # We have to advance the simulation by enough time steps to ensure all actions are executed
        latest_action_time = max((ac.time_step for ac in valid_actions.commands), default=0)
        min_steps = latest_action_time + time_between_actions

        assert self.world is not None, "World is not initialized"

        owner = getattr(self, "outcome_owner", None)
        if owner is None:
            self.world = simulate_n_next_states(
                min_steps,
                SIMULATOR_TIMESTEP_MS,
                self.world,
                valid_actions.to_tuple_list(),  # ty: ignore[invalid-argument-type]
                None,
            )
        else:
            self.world = owner.advance(self.world, min_steps, valid_actions.to_tuple_list())
        valid_actions.apply_side_effects()

    def get_current_route(self, callsign: str) -> tuple[str, list[RoutePoint]]:
        """Gets the current route for an aircraft by its callsign."""

        assert self.world is not None, "World is not initialized"

        aircraft = self.world.get_aircraft_by_callsign(callsign)

        if aircraft.route_name is None:
            runtime_print("Simulator", "warning", message=f"aircraft {callsign} has no route name")

        return aircraft.route_name or "UNKNOWN", aircraft.route

    def _load_configs(
        self, simulation_config_path: SimulationConfigPath, difficulty: MissionDifficulty
    ):
        runtime_print(
            "Simulator",
            "init",
            config_root=simulation_config_path.simulation_config_root,
        )
        if not os.path.exists(simulation_config_path.simulation_config_root):
            raise FileNotFoundError(
                f"Scenario configuration root dir does not exist: "
                f"{simulation_config_path.simulation_config_root}"
            )

        self.scenario_config = load_config(
            simulation_config_path.simulation_config_root, difficulty
        )
        self.designs = self.scenario_config.aircraft_designs
        self.parsed_designs = parse_all_aircraft_designs(dict(self.designs))

    def _sample_start_time_offset_seconds(
        self,
        *,
        seed: int,
        max_offset_s: int | None = None,
    ) -> int:
        """Sample scenario start-time offset in seconds."""
        effective_max = (
            int(self.settings.scenario_start_time_offset_max_s)
            if max_offset_s is None
            else int(max_offset_s)
        )
        effective_max = max(0, effective_max)
        if effective_max == 0:
            return 0

        return random.Random(int(seed)).randint(0, effective_max)

    def _advance_world_with_start_offset(
        self,
        world: World,
        *,
        seed: int,
        max_offset_s: int | None = None,
    ) -> World:
        """Advance the world by a sampled start-time offset."""
        offset_s = self._sample_start_time_offset_seconds(
            seed=seed,
            max_offset_s=max_offset_s,
        )
        if offset_s <= 0:
            return world
        owner = getattr(self, "outcome_owner", None)
        branch = owner.fork() if owner is not None else None
        advanced_world = (
            branch.advance(world, offset_s)
            if branch is not None
            else simulate_n_next_states(offset_s, SIMULATOR_TIMESTEP_MS, world, [])
        )
        # Avoid degenerate starts where all missions have already completed.
        if not self._has_unfinished_arrival_targets(advanced_world):
            return world
        if owner is not None and branch is not None:
            owner.commit(branch, advanced_world)
        return advanced_world

    @staticmethod
    def _has_unfinished_arrival_targets(world: World) -> bool:
        """Return True when at least one arrival target is not completed."""
        arrival_targets = getattr(world, "arrival_targets", [])
        for target in arrival_targets:
            callsign = getattr(target, "ac_callsign", None)
            if callsign is None:
                continue
            aircraft = world.get_aircraft_by_callsign(callsign)
            if aircraft is None:
                return True
            if not bool(getattr(aircraft.mission, "completed", False)):
                return True

        # Conservative fallback for unexpected worlds: if there are aircraft but no
        # arrival-target metadata, avoid rejecting the advanced world by default.
        aircraft_list = getattr(world, "aircraft", [])
        return bool(aircraft_list) and not arrival_targets

    def _find_valid_scenario(
        self,
        difficulty: MissionDifficulty,
        seed: int,
        eval_mode: bool,
        max_aircraft_count: int | None = None,
    ) -> World:
        """Find a valid scenario without infringements."""
        if not self.seed_set:
            self.seed_set = True
            set_seed(seed)

        attempts = 0

        while True:
            attempts += 1
            mission = Mission(
                completed=False,
                failed=False,
                aircraft_horizontal_safezone_radius_nm=self.settings.horizontal_safezone_radius_nm,
                aircraft_vertical_safezone_radius_ft=self.settings.vertical_safezone_radius_ft,
            )

            # Training defaults to the legacy fixed aircraft count when no explicit cap is passed.
            if max_aircraft_count is not None:
                aircraft_count = max_aircraft_count
            else:
                aircraft_count = (
                    int(self.settings.max_aircraft_count_training)
                    if not eval_mode
                    else self.settings.max_aircraft_count_eval
                )

            try:
                # Rust/PyO3 may raise runtime panics for invalid sampled params.
                # Treat them as transient and retry sampling a new scenario.
                assert self.scenario_config is not None, "Scenario config is not loaded"
                mission_parts = generate_random_mission(
                    aircraft_count,
                    self.scenario_config,
                    mission,
                    difficulty == MissionDifficulty.NORMAL,
                    None,
                    difficulty,
                )
                aircraft = mission_parts[0]
                waypoints = mission_parts[1]
                airspace = mission_parts[2]
                arrival_targets = mission_parts[3]
                load_world_untyped = cast(Any, load_world)
                world = load_world_untyped(
                    aircraft,
                    arrival_targets,
                    self.designs,
                    airspace,
                    waypoints,
                )
            except BaseException as err:  # noqa: BLE001
                runtime_print(
                    "Simulator",
                    "warning",
                    message=(
                        f"scenario generation attempt {attempts} failed "
                        f"({type(err).__name__}: {err}); retrying"
                    ),
                )
                continue

            # Cut out and store trombone parts from routes, to be added back by actions.
            if getattr(self, "capture_generation", False):
                self.raw_generated_world_json = world.json()
            world = self._cut_and_store_trombone_from_routes(aircraft, world)

            self.last_scenario_generation_attempts = attempts
            return world

    def _cut_and_store_trombone_from_routes(self, acs: list[Aircraft], world: World) -> World:
        owner = getattr(self, "outcome_owner", None)
        if owner is not None:
            owner.begin(world)
        # Keep store aligned with exactly one scenario/world at a time.
        self.route_trombone_store.clear()
        aircraft_commands: list[AircraftCommand] = []
        for ac in acs:
            cut_route = self.route_trombone_store.cut_out_section(ac.call_sign, ac.route)
            aircraft_commands.append(
                AircraftCommand.SET_ROUTE(  # ty: ignore[call-non-callable]
                    call_sign=ac.call_sign, route=cut_route
                )
            )

        self._build_trombone_exclusion_polygons(acs, world)
        if owner is not None:
            return owner.advance_next(world, 0, aircraft_commands)
        return simulate_next_state(0, world, aircraft_commands, None)

    def _build_trombone_exclusion_polygons(self, acs: list[Aircraft], world: World) -> None:
        """Build one combined convex-hull exclusion polygon from ALL trombone areas.

        Aggregates excluded waypoints from every aircraft's trombone section
        plus the boundary markers (section start/end) into a single polygon.
        Uses world.waypoints to resolve positions — no coordinate duplication.
        Called once per episode right after cutting sections.
        """
        waypoints_map = world.waypoints
        all_positions: list[tuple[float, float]] = []

        for ac in acs:
            excluded_points = self.route_trombone_store.get_all_excluded_points(ac.call_sign)
            if excluded_points is None:
                continue

            for rp in excluded_points:
                pos = resolve_waypoint_position(rp, waypoints_map)
                if pos is not None:
                    all_positions.append(pos)

            # Include boundary markers (section start and end) for a complete hull
            boundaries = self.route_trombone_store.get_section_boundary_names(ac.call_sign)
            if boundaries is not None:
                for name in boundaries:
                    wp = waypoints_map.get(str(name))
                    pos_nm = getattr(wp, "position_nm", None) if wp is not None else None
                    if pos_nm is not None:
                        all_positions.append((float(pos_nm.x), float(pos_nm.y)))

        if len(all_positions) >= 3:
            full_polygon = convex_hull(all_positions)
            exclusion = shrink_polygon(
                full_polygon, factor=self.settings.trombone_exclusion_shrink_factor
            )
            self.route_trombone_store.set_trombone_exclusion_polygon(exclusion)

    def prepare_world_for_episode(self, world: World) -> World:
        """Normalize externally provided worlds to match generated-scenario route preprocessing."""
        result = self._cut_and_store_trombone_from_routes(list(world.aircraft), world)
        owner = getattr(self, "outcome_owner", None)
        self._prepared_world_token = (
            result,
            world_digest(result),
            owner,
            owner.current_identity if owner is not None else None,
        )
        return result

    def reset(
        self,
        difficulty: MissionDifficulty,
        create_world: bool = False,
        world: World | None = None,
        seed: int | None = None,
        reset_seed: bool = False,
        eval_mode: bool = False,
        max_aircraft_count: int | None = None,
        start_time_offset_max_s: int | None = None,
        episode_steps: int | None = None,
        world_prepared: bool = False,
    ) -> World:
        """Reset the simulator to initial state."""

        owner = getattr(self, "outcome_owner", None)
        prepared = getattr(self, "_prepared_world_token", None)
        self._prepared_world_token = None
        if world_prepared:
            if (
                prepared is None
                or world is not prepared[0]
                or world_digest(world) != prepared[1]
                or owner is not prepared[2]
                or (owner.current_identity if owner is not None else None) != prepared[3]
            ):
                raise ValueError("prepared world token is missing, changed or already consumed")
        if reset_seed:
            self.seed_set = False

        self.episode_steps = (
            int(self.settings.episode_steps) if episode_steps is None else int(episode_steps)
        )

        self.recorder = DataRecorder()
        self.recorder.init(
            SIMULATOR_TIMESTEP_MS,  # timestep_s
            0,  # timestamp_start
            (self.settings.origin_lat, self.settings.origin_lon),  # origin_lat_lon_deg
        )

        if world is not None:
            self.last_scenario_generation_attempts = 0
            self.initial_world = world
            if owner is not None and not world_prepared:
                owner.begin(world)
        elif self.settings.world_json_file:
            self.last_scenario_generation_attempts = 0
            self.initial_world = self._load_world_from_json(self.settings.world_json_file)
            if owner is not None:
                owner.begin(self.initial_world)
        elif create_world:
            if seed is None:
                raise ValueError("seed must be set when create_world=True")
            scenario_seed = int(seed)
            self.initial_world = self._find_valid_scenario(
                difficulty, scenario_seed, eval_mode, max_aircraft_count
            )
            self.initial_world = self._advance_world_with_start_offset(
                self.initial_world,
                seed=scenario_seed,
                max_offset_s=start_time_offset_max_s,
            )

        elif owner is not None and self.initial_world is not None:
            owner.begin(self.initial_world)

        if self.initial_world is not None:
            self.world = self.initial_world

        assert self.world is not None, "World is not initialized"

        return self.world

    def _load_world_from_json(self, json_file: str) -> World:
        """Load a World from a JSON file."""
        with open(json_file, "r", encoding="utf-8") as f:
            json_str = f.read()

        world = World.from_json(json_str)
        assert isinstance(world, World)

        return world
