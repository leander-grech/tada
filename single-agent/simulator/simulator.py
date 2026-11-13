"""Module for the simulator"""

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast


from config.config import Config
from config.simulation_config_path import SimulationConfigPath
from flight_simulator import (  # type: ignore[import, attr-defined]
    AircraftDesign,
    Aircraft,
    DataRecorder,
    Mission,
    MissionDifficulty,
    RoutePoint,
    ScenarioConfig,
    World,
    generate_random_mission,
    load_world,
    load_config,
    set_seed,
    simulate_n_next_states,
    simulate_next_state,
    AircraftCommand
)
from models.commands import TimedCommandList
from simulator.design_parser import (
    AircraftDesignData,
    parse_all_aircraft_designs,
)
from utils.infringement_utils import get_world_infringements, is_realised_los
from simulator.route_section_store  import RouteSectionStore
from utils.runtime_log import runtime_print


class Simulator:
    """Simulator class for the environment"""

    def __init__(self,
                 difficulty: MissionDifficulty,
                 scenario_config_path: SimulationConfigPath,
                 route_trombone_utils: RouteSectionStore,
                 verbose: bool = False
                 ):
        """Initialize the simulator with config and load initial state."""
        self.config = Config
        self._verbose = verbose
        self.scenario_config: ScenarioConfig = None
        self.recorder = DataRecorder()
        self.world: Optional[World] = None
        self.initial_world: Optional[World] = None
        self.designs: Mapping[str, AircraftDesign] = {}
        self.parsed_designs: Dict[str, AircraftDesignData] = {}
        self.seed_set = False

        # Number of aircraft to GENERATE per scenario, decoupled from N_AIRCRAFT_MAX (the
        # number of observation/action SLOTS). None => fall back to Config (see
        # _find_valid_scenario). The windowed env sets this to 20 while keeping 10 slots.
        self.scenario_aircraft_count: Optional[int] = None

        # Stitching (windowed env): a scenario of `stitch_segments` independently generated
        # segments of `scenario_aircraft_count` flights each, the next segment's first landing
        # placed a random `stitch_gap_s` after the previous segment's last one. 1 = no stitching.
        self.stitch_segments: int = 1
        self.stitch_gap_s: Tuple[float, float] = (120.0, 900.0)
        self.stitch_gaps_last: List[float] = []  # gaps drawn for the current scenario

        # Record scenario-rejection telemetry.
        self.scenario_attempts_last = 0
        self.scenario_attempts_total = 0
        self.scenarios_accepted = 0

        self.route_trombone_utils = route_trombone_utils
        self._load_configs(scenario_config_path, difficulty)

    @staticmethod
    def advance_multiple_steps(
            # self,
            steps: int,
            world: World,
            time_between_actions: int = Config.TIME_BETWEEN_ACTIONS
    ) -> World:
        """Advance the simulator forward by 'steps' timesteps."""
        return simulate_n_next_states(
            steps * time_between_actions,
            1000,  # In milliseconds, do not modify for now
            world,
            [],
        )

    def apply_action_and_advance(
        self,
        actions: TimedCommandList,
        time_between_actions: int = Config.TIME_BETWEEN_ACTIONS,
    ):
        """
        Apply the given actions, then advance the simulator by TIME_BETWEEN_ACTIONS timesteps.
        """

        valid_actions = TimedCommandList(
            commands=[action for action in actions.commands if action.command is not None]
        )

        # We have to advance the simulation by enough time steps to ensure all actions are executed
        latest_action_time = max((ac.time_step for ac in valid_actions.commands), default=0)
        min_steps = latest_action_time + time_between_actions

        assert self.world is not None, "World is not initialized"

        self.world = simulate_n_next_states(
            min_steps,
            1000,  # In milliseconds, do not modify for now
            self.world,
            valid_actions.to_tuple_list(),  # type: ignore[arg-type]
            None,
        )

    def simulate_candidate(
        self,
        actions: TimedCommandList,
        time_between_actions: int = Config.TIME_BETWEEN_ACTIONS,
    ) -> World:
        """Pure preview of apply_action_and_advance: apply actions and advance one interval
        from the current world WITHOUT committing (does not modify self.world).

        simulate_n_next_states returns a new world and leaves the input untouched, so this is
        safe for read-only look-ahead (the env's peek_action shield).
        """
        assert self.world is not None, "World is not initialized"
        valid_actions = TimedCommandList(
            commands=[action for action in actions.commands if action.command is not None]
        )
        latest_action_time = max((ac.time_step for ac in valid_actions.commands), default=0)
        min_steps = latest_action_time + time_between_actions
        return simulate_n_next_states(
            min_steps,
            1000,
            self.world,
            valid_actions.to_tuple_list(),  # type: ignore[arg-type]
            None,
        )

    def get_current_route(self, callsign: str) -> Tuple[str, List[RoutePoint]]:
        """Gets the current route for an aircraft by its callsign."""

        assert self.world is not None, "World is not initialized"

        aircraft = self.world.get_aircraft_by_callsign(callsign)

        if aircraft.route_name is None:
            print(f"Aircraft {callsign} has no route name")

        return aircraft.route_name or "UNKNOWN", aircraft.route

    def check_not_imminent_infringement(self, world: World) -> bool:
        """Reject scenarios that are already lost at birth.

        True if no pair reaches a realised loss of separation (severity 1.0 == inside 3 NM
        AND 500 ft) within the first SCENARIO_INITIAL_CONFLICT_FREE_STEPS of a do-nothing
        rollout.

        """
        world_advanced = self.advance_multiple_steps(
            steps=self.config.SCENARIO_INITIAL_CONFLICT_FREE_STEPS, world=world
        )
        infringements = get_world_infringements(world_advanced)
        return not any(
            is_realised_los(infringement.value)
            for aircraft_infringements in infringements.values()
            for infringement in aircraft_infringements
        )

    def _load_configs(self, simulation_config_path: SimulationConfigPath, difficulty: MissionDifficulty):
        if self.config.SCENARIO_GENERATOR == "procedure":
            # One frozen generator snapshot per procedure.
            use_case = {MissionDifficulty.VALIDATION_USE_CASE_1: 1, MissionDifficulty.VALIDATION_USE_CASE_2: 2}.get(difficulty)
            if use_case is None:
                raise ValueError(f"TADA_GENERATOR=procedure has generator snapshots for use cases 1 and 2 only, not {difficulty}")
            simulation_config_path = SimulationConfigPath(
                os.path.abspath(self.config.PROCEDURE_CONFIG_ROOTS[use_case]))
        if self._verbose: print(f"Creating simulator, config root: {simulation_config_path.simulation_config_root}")
        if not os.path.exists(simulation_config_path.simulation_config_root):
            raise FileNotFoundError(
                f"Scenario configuration root dir does not exist: {simulation_config_path.simulation_config_root}")

        self.scenario_config = load_config(simulation_config_path.simulation_config_root, difficulty)
        self.designs = self.scenario_config.aircraft_designs
        self.parsed_designs = parse_all_aircraft_designs(self.designs)

    def _find_valid_scenario(
            self,
            difficulty: MissionDifficulty,
            seed: Optional[int], eval_mode: bool,
            max_aircraft_count: Optional[int] = None
        ) -> World:
        """Find a valid scenario without infringements."""
        if seed is not None and not self.seed_set:
            self.seed_set = True
            set_seed(seed)

        attempts = 0

        while True:
            attempts += 1
            mission = Mission(
                completed=False,
                failed=False,
                aircraft_horizontal_safezone_radius_nm=self.config.MISSION_SAFEZONE_RADIUS_NM,
                aircraft_vertical_safezone_radius_ft=self.config.VERTICAL_SAFEZONE_RADIUS_FT,
            )

            # Determine aircraft count: use provided value, or fall back to config
            # Precedence: explicit argument > per-instance override > Config.SCENARIO_AIRCRAFT_COUNT_*
            # (same optional keys as the UM_windowed_approach branch) > MAX_AIRCRAFT_COUNT_*.
            if max_aircraft_count is not None:
                aircraft_count = max_aircraft_count
            elif self.scenario_aircraft_count is not None:
                aircraft_count = int(self.scenario_aircraft_count)
            else:
                aircraft_count = (
                    getattr(self.config, "SCENARIO_AIRCRAFT_COUNT_TRAIN", self.config.MAX_AIRCRAFT_COUNT_TRAIN)
                    if not eval_mode
                    else getattr(self.config, "SCENARIO_AIRCRAFT_COUNT_EVAL", self.config.MAX_AIRCRAFT_COUNT_EVAL)
                )

            try:
                # Rust/PyO3 may raise runtime panics for invalid sampled params.
                # Treat them as transient and retry sampling a new scenario.
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
            # world = load_world(aircraft, self.designs, airspace, waypoints)
            # world = load_world(aircraft, arrival_targets, self.designs, airspace, waypoints)

            # Cut out and store trombone parts from routes, to be added back by actions as necessary
            world = self._cut_and_store_trombone_from_routes(aircraft, world)

            # The procedure generator deals the first scenario a seed draws: the multi-agent simulator
            # has no rejection filter, and a rejection here would make the seed a different episode.
            if self.config.SCENARIO_GENERATOR == "procedure" or self.check_not_imminent_infringement(world):
                self.scenario_attempts_last = attempts
                self.scenario_attempts_total += attempts
                self.scenarios_accepted += 1
                return world

            # Bound rejection attempts so reset fails with a diagnostic.
            if attempts >= int(self.config.SCENARIO_MAX_ATTEMPTS):
                raise RuntimeError(
                    f"no conflict-free scenario after {attempts} attempts "
                    f"(difficulty={difficulty}, seed={seed}, aircraft={aircraft_count}). "
                    "The imminent-infringement filter is rejecting nearly everything — "
                    "check SEVERITY_CONFLICT_NM / SCENARIO_INITIAL_CONFLICT_FREE_STEPS."
                )

    def _cut_and_store_trombone_from_routes(self, acs: list[Aircraft], world: World) -> World:
        aircraft_commands: List[AircraftCommand] = []
        for ac in acs:
            cut_route = self.route_trombone_utils.cut_out_section(ac.call_sign, ac.route)
            aircraft_commands.append(AircraftCommand.SET_ROUTE(
                    call_sign=ac.call_sign,
                    route=cut_route))

        return simulate_next_state(0, world, aircraft_commands, None)

    def reset(
        self,
        difficulty: MissionDifficulty,
        create_world: bool = False,
        world: Optional[World] = None,
        seed: Optional[int] = None,
        reset_seed: bool = False,
        eval_mode: bool = False,
    ) -> World:
        """Reset the simulator to initial state."""

        if reset_seed:
            self.seed_set = False

        self.recorder = DataRecorder()
        self.recorder.init(
            1000,  # timestep_s
            0,  # timestamp_start
            (self.config.ORIGIN_LAT, self.config.ORIGIN_LON),  # origin_lat_lon_deg
        )

        if world is not None:
            self.initial_world = world
        elif self.config.WORLD_JSON_FILE:
            self.initial_world = self._load_world_from_json(self.config.WORLD_JSON_FILE)
        elif create_world and int(self.stitch_segments) > 1:
            self.initial_world = self._stitched_scenario(difficulty, seed, eval_mode)
        elif create_world:
            self.initial_world = self._find_valid_scenario(difficulty, seed, eval_mode)

        if self.initial_world is not None:
            self.world = self.initial_world

        assert self.world is not None, "World is not initialized"

        return self.world

    @staticmethod
    def _renumber(call_sign: str, offset: int) -> str:
        """'TEST005' + 20 -> 'TEST025': keeps the prefix and the queue-number convention."""
        prefix = call_sign.rstrip("0123456789")
        digits = call_sign[len(prefix):]
        width = max(3, len(digits))
        return f"{prefix}{int(digits or 0) + offset:0{width}d}"

    def _stitched_scenario(
            self, difficulty: MissionDifficulty, seed: Optional[int], eval_mode: bool
    ) -> World:
        """A long arrival stream built from short, independently generated segments.

        Generating N*20 flights in one sequence lets the generator's schedule pressure build
        along the queue (do-nothing |dev| grows from ~220 s to ~910 s over 40 flights), so a
        long stream tests backlog rather than steady-state control. Stitching generates each
        segment on its own -- same distribution as the 20-flight training scenarios -- and
        places segment k+1 so that its FIRST target landing comes a random cooling gap
        (uniform in ``stitch_gap_s``) after segment k's LAST one. Spawn times and targets of the
        segment move by the same offset, so its internal spacing is untouched; its callsigns
        are renumbered to continue the queue (TEST021...), and their trombone sections re-keyed.

        Deterministic for a pinned ``seed``: the segments come from one seeded generator stream,
        and the gaps from an RNG seeded with it.
        """
        import json
        import random as _random

        from flight_simulator import World as _World

        rng = _random.Random(seed) if seed is not None else _random.Random()
        store = self.route_trombone_utils._route_parts
        saved_sections = {}
        merged = None
        last_eta = 0.0
        self.stitch_gaps_last = []
        for _ in range(int(self.stitch_segments)):
            segment = json.loads(self._find_valid_scenario(difficulty, seed, eval_mode).json())
            if merged is None:
                merged = segment
                for ac in segment["aircraft"]:
                    saved_sections[ac["call_sign"]] = store[ac["call_sign"]]
            else:
                gap = float(rng.uniform(*self.stitch_gap_s))
                self.stitch_gaps_last.append(gap)
                offset = last_eta + gap - min(float(t["eta_s"]) for t in segment["arrival_targets"])
                assert offset > 0.0, f"stitch offset {offset:.0f} s would move spawns before t=0"
                rename = {ac["call_sign"]: self._renumber(ac["call_sign"], len(merged["aircraft"]))
                          for ac in segment["aircraft"]}
                for ac in segment["aircraft"]:
                    # The segment was just generated, so the store holds ITS sections under the
                    # old names (it overwrote the previous segment's -- restored below).
                    saved_sections[rename[ac["call_sign"]]] = store[ac["call_sign"]]
                    ac["call_sign"] = rename[ac["call_sign"]]
                    ac["start_time_s"] = float(ac["start_time_s"]) + offset
                for t in segment["arrival_targets"]:
                    t["ac_callsign"] = rename[t["ac_callsign"]]
                    t["eta_s"] = float(t["eta_s"]) + offset
                merged["aircraft"] += segment["aircraft"]
                merged["arrival_targets"] += segment["arrival_targets"]
                merged["waypoints"].update(segment["waypoints"])
            last_eta = max(float(t["eta_s"]) for t in merged["arrival_targets"])
        store.update(saved_sections)
        return _World.from_json(json.dumps(merged))

    def _load_world_from_json(self, json_file: str) -> World:
        """Load a World from a JSON file."""
        with open(json_file, "r", encoding="utf-8") as f:
            json_str = f.read()

        world = World.from_json(json_str)  # type: ignore[attr-defined]
        assert isinstance(world, World)

        return world
