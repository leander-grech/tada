"""Fail-closed native identity and actual owned actor activation checks."""

from __future__ import annotations

import os
import sys
from functools import partial

import ray
import torch
import torch.version

from common.hashing import content_digest
from common.sealed import consume
from training.frozen_settings import require_initial_preprocessing_window
from training.reservations import effective_reserved_seeds
from training.run_config import ResolvedRunConfig
from training.seed_stream import ScenarioSeedStream

SUPPORTED_RAY = "2.53.0"
_NATIVE_FIELDS = (
    "mode",
    "version",
    "source_identity",
    "source_provenance",
    "wheel_sha256",
    "extension_sha256",
)


def verify_native_artifact(conditions: dict) -> dict:
    """Authenticate actual imported bytes, including for direct bound environments."""
    from common.native_runtime import verify_native

    actual = verify_native()
    expected = conditions["native_artifact"]
    if any(key not in expected or actual[key] != expected[key] for key in _NATIVE_FIELDS):
        raise ValueError("native artifact mismatch against frozen reservation conditions")
    return actual


def verify_hash_policy() -> dict:
    """Reject late environment mutation as well as randomized interpreter startup."""
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
        raise ValueError(
            "effective Python hash policy requires PYTHONHASHSEED=0 before interpreter startup"
        )
    return {"version": 1, "seed": 0, "hash_randomization": 0}


def verify_native_conditions(run: ResolvedRunConfig) -> dict:
    """Verify actual native bytes, frozen preprocessing and supported runtime."""
    verify_hash_policy()
    actual = verify_native_artifact(run.reserved_conditions)
    require_initial_preprocessing_window(run.thresholds, run.reserved_conditions)
    backend = "cuda" if torch.version.cuda is not None else "cpu"
    if backend != run.resources["backend"]:
        raise ValueError("native runtime backend mismatch against resolved resources")
    if ray.__version__ != SUPPORTED_RAY:
        raise ValueError("unsupported Ray runtime for owned learning")
    return actual


def _class_name(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def verify_owned_class(actual_type, expected, role):
    """Accept only the owned class or Ray's exact actor wrapper layout."""
    from ray.train._internal.worker_group import RayTrainWorker

    original = actual_type.__dict__.get("__ray_actor_class__")
    ray_wrapper = (
        original is not None
        and actual_type.__bases__ == (original,)
        and (
            original is expected
            or (
                role == "learner"
                and original.__module__ == "ray.train._internal.worker_group"
                and original.__name__ == "_WrappedExecutable"
                and original.__bases__ == (expected, RayTrainWorker)
            )
        )
    )
    if actual_type is not expected and not ray_wrapper:
        raise ValueError(f"actual {role} class is not the owned {expected.__name__}")
    return ray_wrapper


def _inspect_actor(actor, *, run: ResolvedRunConfig, role: str) -> dict:
    from atc_env.sector_selection import capture_loaded_catalog_identity
    from network.learner_connectors import _EXPECTED_PIPELINE, ATCGeneralAdvantageEstimation
    from network.rlm import ATCRLM
    from training.env import TrainingATCEnv
    from training.env_runner import ATCEnvRunner
    from training.learner_checkpoint import ATCPPOLearner

    catalog_identity = capture_loaded_catalog_identity()
    expected = ATCPPOLearner if role == "learner" else ATCEnvRunner
    actual_type = type(actor)
    ray_wrapper = verify_owned_class(actual_type, expected, role)
    native = verify_native_conditions(run)
    modules = []
    if actor.module is None:
        raise ValueError("actual owned actor has no model")
    for module_id in actor.module.keys():
        module = actor.module[module_id].unwrapped()
        if type(module) is not ATCRLM:
            raise ValueError("actual module class is not owned ATCRLM")
        consume(module.model_config["resolved_model"], run.semantic_digest, role="model")
        if module.consumed_semantic_digest != run.semantic_digest:
            raise ValueError("actual module consumption mismatch")
        modules.append(
            {
                "module_id": module_id,
                "class": _class_name(module),
                "semantic_digest": module.consumed_semantic_digest,
                "device": str(next(module.parameters()).device),
                "model_payload": module.model_config["resolved_model"],
            }
        )
    if len(modules) != 1:
        raise ValueError("owned PPO requires one shared model")
    report = {
        "python_hash_policy": verify_hash_policy(),
        "role": role,
        "class": _class_name(actor),
        "pid": os.getpid(),
        "actor_id": (
            str(ray.get_runtime_context().get_actor_id())
            if role == "learner" or actor.worker_index > 0
            else None
        ),
        "assigned_resources": (
            ray.get_runtime_context().get_assigned_resources()
            if role == "learner" or actor.worker_index > 0
            else {}
        ),
        "owned_class": f"{expected.__module__}.{expected.__qualname__}",
        "ray_actor_wrapper": ray_wrapper,
        "native": native,
        "ray": ray.__version__,
        "modules": modules,
        "semantic_digest": run.semantic_digest,
        "catalog": catalog_identity,
    }
    from dataclasses import asdict

    from training.resource_profile import observe_actor_device

    report["device_report"] = None
    if report["actor_id"] is not None:
        report["device_report"] = asdict(
            observe_actor_device(
                role,
                report["actor_id"],
                next(actor.module[modules[0]["module_id"]].parameters()).device,
            )
        )
    else:
        # The driver holds a local model for synchronization, never a sampling
        # environment. Its checkpoint RNG capture may initialize driver CUDA.
        if actor.env is not None or actor.config.num_env_runners < 1:
            raise ValueError("unreported local sampling actor is unsupported")
        report["driver_device"] = {
            "sampling_environment": False,
            "cuda_initialized": torch.cuda.is_initialized(),
        }

    if role == "learner":
        connectors = actor._learner_connector.connectors
        names = tuple(type(c).__name__ for c in connectors)
        if (
            names != (*_EXPECTED_PIPELINE[:-1], "ATCGeneralAdvantageEstimation")
            or type(connectors[-1]) is not ATCGeneralAdvantageEstimation
            or any(
                not type(c).__module__.startswith("ray.rllib.connectors.") for c in connectors[:-1]
            )
        ):
            raise ValueError("actual learner connector stack is not the owned supported GAE layout")
        if (
            connectors[-1].gamma != run.optimizer["gamma"]
            or connectors[-1].lambda_ != run.optimizer["gae_lambda"]
        ):
            raise ValueError("actual GAE settings mismatch")
        if modules[0]["device"].split(":")[0] != run.resources["learner_device"]:
            raise ValueError("actual learner device mismatch")
        if (
            actor.config.lr != run.optimizer["learning_rate"]
            or actor.config.entropy_coeff != run.optimizer["entropy_coeff"]
            or actor.config.use_kl_loss
        ):
            raise ValueError("actual learner constant optimizer settings mismatch")
        report["connectors"] = [_class_name(c) for c in connectors]
        report["optimizer"] = {
            "learning_rate": actor.config.lr,
            "entropy_coeff": actor.config.entropy_coeff,
            "use_kl_loss": actor.config.use_kl_loss,
        }
    else:
        from atc_env.episode import ATCEpisode, validate_episode_adapter
        from training.env_runner import _OWNED_NEW_EPISODE

        validate_episode_adapter(actor, _OWNED_NEW_EPISODE)
        report["episode_adapter"] = f"{ATCEpisode.__module__}.{ATCEpisode.__qualname__}"
        report["worker_index"] = actor.worker_index
        report["environments"] = []
        if actor.env is not None:
            for vector_index, wrapped in enumerate(actor.env.envs):
                env = wrapped.unwrapped
                if type(env) is not TrainingATCEnv or not env._runtime_activation_required:
                    raise ValueError("actual environment requires the owned activation guard")
                consume(env._training_payload, run.semantic_digest, role="env")
                if not env._runtime_activated and (
                    env.simulator.world is not None or env._has_reset_once
                ):
                    raise ValueError(
                        "native generation occurred before actual selection verification"
                    )
                state = _verify_environment_identity(env, actor, run, role, vector_index)
                report["environments"].append(
                    {
                        "semantic_digest": env.consumed_semantic_digest,
                        "evaluation_mode": env.evaluation_mode,
                        "activated": env._runtime_activated,
                        "success_time_dev_s": env.success_time_dev_s,
                        "stream": state,
                        "vector_index": vector_index,
                        "features": env._training_payload["features"],
                        "reward": env.reward_settings,
                        "scenario": env._training_payload["scenario"],
                    }
                )
    return report


def _verify_environment_identity(env, actor, run, role, vector_index):
    """Inspect effective instance identity, not merely its original payload."""
    evaluation = role == "operational_evaluation"
    if env.evaluation_mode is not evaluation or bool(actor.config.in_evaluation) != evaluation:
        raise ValueError("actual environment role/mode mismatch")
    threshold = run.thresholds["operational_success_s" if evaluation else "training_success_s"]
    if env.success_time_dev_s != threshold or (evaluation and threshold != 60):
        raise ValueError("actual environment threshold mismatch")
    if (env._worker_index, env._vector_index) != (actor.worker_index, vector_index):
        raise ValueError("actual environment runner/vector identity mismatch")
    reserved = effective_reserved_seeds(run.reserved_manifest)
    expected = ScenarioSeedStream(
        run.root_seed,
        (run.topology["phase_slot"], actor.worker_index, vector_index),
        reserved,
        domain=role,
    )
    state = env.get_training_stream_state()
    if (
        state["semantic_digest"] != run.semantic_digest
        or state["reservation_sha256"] != run.reserved_manifest["manifest_sha256"]
        or state["conditions_sha256"] != run.reserved_conditions["conditions_sha256"]
        or env._training_stream.reserved_seeds != reserved
    ):
        raise ValueError("actual environment stream reservation identity mismatch")
    actual_identity = {
        key: value for key, value in state["stream"].items() if key not in {"cursor", "rejected"}
    }
    expected_identity = {
        key: value
        for key, value in expected.get_state().items()
        if key not in {"cursor", "rejected"}
    }
    if content_digest(actual_identity) != content_digest(expected_identity):
        raise ValueError("actual environment stream identity mismatch")
    try:
        expected.set_state(state["stream"])
    except ValueError as error:
        raise ValueError("actual environment stream state mismatch") from error
    if not env._runtime_activated:
        # A validated closed 10A boundary carries prior progress. A retained
        # journal may belong to that prior actor; file presence is not evidence
        # that this guarded actor generated anything. Pristine boundaries still
        # require cursor zero and no native world.
        try:
            boundary = env.capture_boundary_state()
            env.restore_boundary_state(boundary, validate_only=True)
        except ValueError as error:
            raise ValueError("actual environment stream advanced before activation") from error
    return state


def _runner_groups(algorithm):
    groups = [
        ("training", algorithm.env_runner_group),
        ("operational_evaluation", algorithm.eval_env_runner_group),
    ]
    if any(group is None for _, group in groups):
        raise ValueError("owned PPO requires training and operational evaluation runner groups")
    return groups


def verify_algorithm_selection(algorithm, run: ResolvedRunConfig) -> dict:
    """Inspect actual processes; availability or a supplied boolean is never proof."""
    from training.algorithm import ATCPPO

    if type(algorithm) is not ATCPPO:
        raise ValueError("actual Algorithm class is not owned ATCPPO")
    reports = []
    for role, group in _runner_groups(algorithm):
        items = group.foreach_env_runner(
            partial(_inspect_actor, run=run, role=role), healthy_only=False
        )
        expected_count = run.topology["env_runners"] if role == "training" else 1
        remote = [item for item in items if item["worker_index"] > 0]
        if (
            len(remote) != expected_count
            or {item["worker_index"] for item in remote} != set(range(1, expected_count + 1))
            or any(len(item["environments"]) != 1 for item in remote)
        ):
            raise ValueError("actual runner topology mismatch or missing actor reports")
        reports.extend(items)
    learners = []
    for result in algorithm.learner_group.foreach_learner(
        partial(_inspect_actor, run=run, role="learner"), healthy_only=False
    ):
        if not result.ok:
            raise result.get()
        learners.append(result.get())
    if len(learners) != run.resources["num_learners"]:
        raise ValueError("actual learner topology mismatch")
    reports.extend(learners)
    remote_actors = [
        item for item in reports if item["role"] == "learner" or item["worker_index"] > 0
    ]
    if len({item["actor_id"] for item in remote_actors}) != len(remote_actors):
        raise ValueError("actual actor identity assignments must be distinct")
    stream_ids = [
        (env["stream"]["stream"]["domain"], *env["stream"]["stream"]["stream_id"])
        for item in reports
        for env in item.get("environments", ())
    ]
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("actual stream identity assignments must be distinct")
    from atc_env.sector_catalog import capture_sector_catalog_identity
    from training.checkpoint_manifest import portable_catalog

    expected_catalog = portable_catalog(capture_sector_catalog_identity())
    if any(item["catalog"] != expected_catalog for item in reports):
        raise ValueError("actual actor catalog identity mismatch")
    from dataclasses import asdict

    from training.resource_profile import (
        ActorDeviceReport,
        resolve_resource_profile,
        runtime_inventory,
        validate_actor_devices,
    )

    profile = getattr(algorithm, "_requested_resource_profile", None) or resolve_resource_profile(
        run, runtime_inventory()
    )
    validate_actor_devices(
        profile, tuple(ActorDeviceReport(**item["device_report"]) for item in remote_actors)
    )
    return {
        "resource_profile": asdict(profile),
        "rollout_backend": run.resources.get("rollout_backend", "scalar"),
        "algorithm": _class_name(algorithm),
        "selection_verified": True,
        "actors": reports,
    }


def _release_runner(runner, *, semantic_digest):
    runner.activate_environments(semantic_digest)


def activate_algorithm(algorithm, run: ResolvedRunConfig) -> dict:
    """Release only after every live actor has passed inspection in this call."""
    proof = verify_algorithm_selection(algorithm, run)
    for _, group in _runner_groups(algorithm):
        group.foreach_env_runner(
            partial(_release_runner, semantic_digest=run.semantic_digest), healthy_only=False
        )
    return proof
