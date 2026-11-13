"""Native input capture and effective controller replay authentication."""

import json
from dataclasses import asdict
from pathlib import Path

from common.hashing import content_digest
from evaluation.benchmark.access import authorize_development
from evaluation.benchmark.contract import BenchmarkContract, procedure_contract, validate_manifest
from evaluation.checkpoint_adapter import InferenceCheckpointEnvMixin
from training.env import TrainingATCEnv


def environment_options(contract, seeds):
    from config.simulation_config_path import SimulationConfigPath
    from evaluation.benchmark.use_cases import procedure_root, selection

    campaign, count = selection(contract.compatibility)
    from evaluation.benchmark.generator_variant import variant_name, variant_root

    root = procedure_root(campaign)
    if variant_name(contract.compatibility) is not None:
        root = variant_root(contract.compatibility["generator_variant"])
    baseline = contract.compatibility["conditions"]["benchmark"]
    stratum = baseline["stratum"]
    return {
        "evaluation_mode": True,
        "evaluation_seeds": list(seeds),
        "max_aircraft_count": count,
        "difficulty": stratum["difficulty"],
        "simulation_config_path": SimulationConfigPath(str(root)),
        "airport": stratum["airport"],
        "max_steps_per_episode": baseline["episode_steps"],
        "scenario_start_time_offset_max_s": baseline["start_offset_max_s"],
        "success_time_dev_s": contract.compatibility["threshold_s"],
        "noop_verification_max_seconds": contract.compatibility["verification_cap_s"],
    }


def _point(point):
    position = point.position_nm
    return {
        "waypoint_name": point.waypoint_name,
        "min_altitude_ft": point.min_altitude_ft,
        "max_altitude_ft": point.max_altitude_ft,
        "target_speed": point.target_speed,
        "position_nm": None if position is None else [position.x, position.y],
    }


def controller_state(env):
    store = env._route_trombone_store
    routes = {}
    for callsign, extender in sorted(store._route_parts.items()):
        routes[callsign] = {
            "kind": type(extender._excluded).__name__,
            "excluded": [_point(p) for p in extender.peek_all_excluded_points()],
            "boundaries": list(extender.boundary_names),
            "original_before": extender._original_last_point_before_excluded,
            "retraction_log": [sorted(level) for level in extender._retraction_log],
        }
    return {
        "route_store": {"routes": routes, "exclusion_polygon": store._trombone_exclusion_polygon},
        "sector_catalog": asdict(env._resolve_selected_group_index()),
    }


def validate_replay(expected, observed):
    left, right = dict(expected), dict(observed)
    left.pop("world_id", None)
    right.pop("world_id", None)
    if content_digest(left) != content_digest(right):
        raise ValueError("benchmark world/effective controller replay identity mismatch")


class MaterializingEnv(TrainingATCEnv):
    """Capture preparation only; reset does not execute a policy or verifier."""

    def reset(self, **kwargs):
        self.simulator.capture_generation = True
        return super().reset(**kwargs)

    def _prepare_initial_rollout(self):
        assert self.simulator.world is not None
        raw = json.loads(self.simulator.raw_generated_world_json)
        initial = json.loads(self.simulator.world.json())
        prepared = super()._prepare_initial_rollout()
        assert self.simulator.world is not None
        state = {
            "seed": self.scenario_seed,
            "raw_world": raw,
            "pre_auto_solve_world": initial,
            "policy_start_world": json.loads(self.simulator.world.json()),
            "effective_controller": controller_state(self),
            "targets": [asdict(target) for target in self.arrival_targets],
            "preprocessing_actions": [dict(a) for a in self._action_history],
            "generation_attempts": self.simulator.last_scenario_generation_attempts,
        }
        state["world_id"] = content_digest(state)
        self.benchmark_world = state
        return prepared


class BenchmarkEnv(InferenceCheckpointEnvMixin, MaterializingEnv):
    """Explicit checkpoint reader subclass, never a training-resume exception."""

    allows_use_case_rebind = True  # diagnostic cross-use-case worlds; recorded in provenance

    def __init__(self, config=None):
        worker = getattr(config, "worker_index", None)
        vector = getattr(config, "vector_index", None)
        config = dict(config or {})
        if worker is not None:
            config["worker_index"] = worker
        if vector is not None:
            config["vector_index"] = vector
        self.benchmark_manifest = validate_development_manifest(config.pop("benchmark_manifest"))
        self.generator_variant = self.benchmark_manifest["contract"]["compatibility"].get(
            "generator_variant"
        )
        super().__init__(config)

    def _journal_training_scenario(self, status):
        if self.generator_variant is None:
            return super()._journal_training_scenario(status)
        from training.provenance import append_scenario_event

        event = {
            **self.get_training_stream_state(),
            "status": status,
            "seed": self.scenario_seed,
            "purpose": "authenticated-development-generator-diagnostic",
            "generator_variant": self.generator_variant,
            "scenario_config_sha256": self.generator_variant["config_sha256"],
            "saved_scenario_config_sha256": self._training_payload["scenario"]["config_sha256"],
            "manifest_sha256": self.benchmark_manifest["manifest_sha256"],
        }
        if status in {"generated", "consumed"} and self.simulator.world is not None:
            event["world_sha256"] = content_digest(json.loads(self.simulator.world.json()))
        append_scenario_event(self._scenario_artifact_path, event)

    def _configure_training_contract(self, config):
        from evaluation.benchmark.binding import benchmark_aircraft_count

        super()._configure_training_contract(config)
        # The manifest, not the checkpoint, selects the generator: a checkpoint trained on a
        # variant replays parent worlds, and the parent replays authenticated variant worlds.
        if self.generator_variant is None:
            self._scenario_config_root = Path(self._training_payload["scenario"]["config_root"])
        else:
            from evaluation.benchmark.generator_variant import variant_root

            self._scenario_config_root = variant_root(self.generator_variant)
        if not self.evaluation_mode or not config.get("inference_checkpoint"):
            raise ValueError("benchmark density requires authenticated checkpoint inference")
        self.max_aircraft_count = benchmark_aircraft_count(
            self._training_payload, self.benchmark_manifest["contract"]["compatibility"]
        )

    def _development_gate(self):
        from evaluation.benchmark.use_cases import selection

        campaign, _ = selection(self.benchmark_manifest["contract"]["compatibility"])
        declared = {row["seed"] for row in self.benchmark_manifest["worlds"]}

        def gate(seeds):
            authorize_development(seeds, campaign=campaign)
            if not set(seeds) <= declared:
                raise ValueError("seed missing from authenticated development manifest")

        return gate

    def reset(self, **kwargs):
        request = getattr(self, "benchmark_request", None)
        if request is not None:
            import random

            import numpy as np

            random.seed(request["rng"]["frame"])
            np.random.seed(request["rng"]["frame"])
        result = super().reset(**kwargs)
        expected = next(
            (row for row in self.benchmark_manifest["worlds"] if row["seed"] == self.scenario_seed),
            None,
        )
        if expected is None:
            raise ValueError("world absent from development manifest")
        validate_replay(expected, self.benchmark_world)
        self.inference_input_binding = {
            "purpose": "authenticated_development",
            "manifest_sha256": self.benchmark_manifest["manifest_sha256"],
            "world_id": expected["world_id"],
        }
        if request is not None:
            import torch

            torch.manual_seed(request["rng"]["policy"])
        return result


def validate_development_manifest(manifest):
    validate_manifest(manifest)
    contract = BenchmarkContract(**manifest["contract"])
    from evaluation.benchmark.use_cases import selection

    campaign, count = selection(contract.compatibility)
    from evaluation.benchmark.generator_variant import variant_name

    procedure_contract(
        "development",
        campaign=campaign,
        aircraft_count=count,
        generator_variant=variant_name(contract.compatibility),
    ).require_compatible(contract)
    authorize_development([world["seed"] for world in manifest["worlds"]], campaign=campaign)
    return manifest
