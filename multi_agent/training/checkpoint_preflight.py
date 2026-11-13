"""Nonmutating preflight against actual constructed modules and learner slots."""

from functools import partial

import numpy as np
import torch
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from common.hashing import content_digest
from training.checkpoint_boundary import CheckpointBoundaryError, restore_boundary_state


def _tensor_layout(saved, actual, path="module"):
    if isinstance(actual, dict):
        if not isinstance(saved, dict) or set(saved) != set(actual):
            raise CheckpointBoundaryError(f"missing or unknown {path} components")
        for key in actual:
            _tensor_layout(saved[key], actual[key], f"{path}/{key}")
    elif isinstance(actual, (torch.Tensor, np.ndarray)):
        if actual.ndim == 0 and type(saved) in (int, float):
            target = (
                torch.empty((), dtype=actual.dtype, device="cpu").numpy()
                if isinstance(actual, torch.Tensor)
                else actual
            )
            with np.errstate(over="ignore", invalid="ignore"):
                converted = np.asarray(saved, dtype=target.dtype)
            if not np.isfinite(saved) or not np.isfinite(converted).all():
                raise CheckpointBoundaryError(f"nonfinite {path} scalar state")
            if target.dtype.kind in "iu" and type(saved) is not int:
                raise CheckpointBoundaryError(f"invalid integer {path} scalar state")
            return
        if not isinstance(saved, (torch.Tensor, np.ndarray)):
            raise CheckpointBoundaryError(f"invalid {path} tensor")
        left = saved.detach().cpu().numpy() if isinstance(saved, torch.Tensor) else saved
        dtype = (
            torch.empty((), dtype=actual.dtype, device="cpu").numpy().dtype
            if isinstance(actual, torch.Tensor)
            else actual.dtype
        )
        if left.shape != tuple(actual.shape) or left.dtype != dtype or not np.isfinite(left).all():
            raise CheckpointBoundaryError(f"incompatible {path} tensor shape/dtype/state")
    elif type(saved) is not type(actual):
        raise CheckpointBoundaryError(f"incompatible {path} state type")


def _metrics(state):
    if not isinstance(state, dict) or set(state) != {"stats"}:
        raise CheckpointBoundaryError("invalid metrics component")
    MetricsLogger().set_state(state)


def validate_learner_state(learner, state):
    """Use the existing strict Adam preflight against live parameter layouts."""
    actual = learner.get_state()
    if not isinstance(state, dict) or set(state) != set(actual):
        raise CheckpointBoundaryError("missing or unknown full learner components")
    actual_modules = {
        module_id: learner.module[module_id].unwrapped().state_dict()
        for module_id in learner.module.keys()
    }
    _tensor_layout(state["rl_module"], actual_modules)
    if state["should_module_be_updated"] != actual["should_module_be_updated"]:
        raise CheckpointBoundaryError("incompatible learner module selection")
    if type(state["weights_seq_no"]) is not int or state["weights_seq_no"] < 0:
        raise CheckpointBoundaryError("invalid learner weight sequence")
    _metrics(state["metrics_logger"])
    learner.validate_checkpoint_state(state)


def _validate_learner_report(learner, *, state):
    try:
        validate_learner_state(learner, state)
    except Exception as error:
        return error
    return None


def preflight_checkpoint(algorithm, state, boundary, run, *, transition=None):
    actual = algorithm.get_state()
    if not isinstance(state, dict) or set(state) != set(actual):
        raise CheckpointBoundaryError("missing or unknown full Algorithm components")
    if type(state["training_iteration"]) is not int or state["training_iteration"] < 0:
        raise CheckpointBoundaryError("invalid lifetime Algorithm iteration")
    _metrics(state["metrics_logger"])
    for key in set(actual) - {"learner_group", "metrics_logger", "training_iteration"}:
        # Supported runner connectors have no evolving component state; their
        # evolving metrics and receipt counters live in the process sidecars.
        _tensor_layout(state[key], actual[key], key)
    if not isinstance(state["learner_group"], dict) or set(state["learner_group"]) != {"learner"}:
        raise CheckpointBoundaryError("unsupported learner group components")
    learner_state = state["learner_group"]["learner"]
    if learner_state.get("atc_effective_state") != boundary["processes"]["learner/0"]["effective"]:
        raise CheckpointBoundaryError("learner and process effective state mismatch")
    reports = algorithm.learner_group.foreach_learner(
        partial(_validate_learner_report, state=learner_state), healthy_only=False
    )
    results = list(reports)
    if len(results) != 1:
        raise CheckpointBoundaryError("missing preflight learner")
    for result in results:
        if not result.ok:
            raise result.get()
        if result.get() is not None:
            raise result.get()
    restore_boundary_state(algorithm, boundary, run, validate_only=True, transition=transition)


def validate_payload_structure(state, boundary, run):
    """Reject absent/unknown mandatory components before Algorithm construction."""
    required = {"learner_group", "metrics_logger", "training_iteration"}
    allowed = required | {
        "env_runner",
        "eval_env_runner",
        "env_to_module_connector",
        "module_to_env_connector",
    }
    if not isinstance(state, dict) or not required <= set(state) or set(state) - allowed:
        raise CheckpointBoundaryError("missing or unknown Algorithm payload components")
    group = state["learner_group"]
    if not isinstance(group, dict) or set(group) != {"learner"}:
        raise CheckpointBoundaryError("missing or unknown learner group")
    learner = group["learner"]
    if not isinstance(learner, dict) or set(learner) != {
        "rl_module",
        "optimizer",
        "metrics_logger",
        "weights_seq_no",
        "should_module_be_updated",
        "atc_effective_state",
    }:
        raise CheckpointBoundaryError("missing or unknown full learner state")
    if (
        not isinstance(learner["rl_module"], dict)
        or not learner["rl_module"]
        or not learner["optimizer"]
    ):
        raise CheckpointBoundaryError("missing module/normalizer/optimizer component")
    if not isinstance(boundary, dict) or set(boundary) != {
        "version",
        "semantic_digest",
        "processes",
    }:
        raise CheckpointBoundaryError("unknown process boundary sidecar")
    if boundary["version"] != 1:
        raise CheckpointBoundaryError("unknown boundary payload version")
    expected = {"driver", "learner/0", "operational_evaluation/1"} | {
        f"training/{i}" for i in range(1, run.topology["env_runners"] + 1)
    }
    if not isinstance(boundary["processes"], dict) or set(boundary["processes"]) != expected:
        raise CheckpointBoundaryError("missing or unknown process sidecar")
    for identity, process in boundary["processes"].items():
        fields = (
            {"rng", "local_runners"}
            if identity == "driver"
            else {"rng", "effective"}
            if identity == "learner/0"
            else {"rng", "runner"}
        )
        if not isinstance(process, dict) or set(process) != fields:
            raise CheckpointBoundaryError("unknown mandatory process payload")
        rng = process["rng"]
        if (
            not isinstance(rng, dict)
            or set(rng) != {"version", "python", "numpy", "torch_cpu", "torch_cuda"}
            or rng["version"] != 1
        ):
            raise CheckpointBoundaryError("unknown mandatory RNG payload")
    from training.checkpoint_envelopes import validate_local_facades, validate_runner_envelope

    validate_local_facades(boundary["processes"]["driver"]["local_runners"])
    for identity in expected - {"driver", "learner/0"}:
        runner = boundary["processes"][identity]["runner"]
        validate_runner_envelope(runner, identity=identity, has_environment=True)
        envelope = runner["environment"]["stream"]
        role, worker = identity.split("/")
        stream = envelope["stream"]
        from training.reservations import effective_reserved_seeds

        reserved = effective_reserved_seeds(run.reserved_manifest)
        if (
            stream["reserved_manifest_hash"] != content_digest(sorted(reserved))
            or envelope["semantic_digest"] != boundary["semantic_digest"]
            or envelope["reservation_sha256"] != run.reserved_manifest["manifest_sha256"]
            or envelope["conditions_sha256"] != run.reserved_conditions["conditions_sha256"]
            or stream["root"] != run.root_seed
            or stream["domain"] != role
            or stream["stream_id"] != [run.topology["phase_slot"], int(worker), 0]
        ):
            raise CheckpointBoundaryError("saved scenario stream identity mismatch")
    effective = learner["atc_effective_state"]
    if (
        not isinstance(effective, dict)
        or set(effective) != {"version", "modules", "learned", "updates", "optimizer_work"}
        or effective["version"] != 2
        or effective != boundary["processes"]["learner/0"]["effective"]
    ):
        raise CheckpointBoundaryError("incompatible effective learner payload")


def module_payload_layout(state):
    """Describe exported model components for integrity preflight before construction."""
    from training.ppo_config import POLICY_ID

    modules = state.get("learner_group", {}).get("learner", {}).get("rl_module")
    if not isinstance(modules, dict) or set(modules) != {POLICY_ID}:
        raise CheckpointBoundaryError("missing or unknown model payload")
    layout = {}
    for module_id, values in modules.items():
        required = {
            "value_normalizer.mean_acc",
            "value_normalizer.sq_mean_acc",
            "value_normalizer.count",
        }
        if not isinstance(values, dict) or not required < set(values):
            raise CheckpointBoundaryError("missing model/normalizer payload")
        layout[module_id] = {}
        for name, value in values.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                if array.dtype.kind not in "bifu" or not np.isfinite(array).all():
                    raise CheckpointBoundaryError("invalid model payload tensor")
                descriptor = {"shape": list(array.shape), "dtype": str(array.dtype)}
            elif type(value) in (int, float) and np.isfinite(value):
                descriptor = {"scalar": type(value).__name__}
            else:
                raise CheckpointBoundaryError("invalid model payload component")
            layout[module_id][name] = descriptor
    return layout


def policy_module_spec(run):
    """The shared policy's module spec with the run's observation and action spaces."""
    import gymnasium as gym

    from config.config import Config
    from models.observation_settings import ObservationSettings
    from models.observation_space import build_observation_space
    from training.ppo_config import POLICY_ID, build_ppo_config

    spec = build_ppo_config(run).rl_module_spec.rl_module_specs[POLICY_ID]
    spec.observation_space = build_observation_space(
        ObservationSettings.from_features(run.features)
    )
    spec.action_space = gym.spaces.Discrete(Config.NUM_ACTIONS)
    return spec


def validate_current_model_payload(state, run):
    """Build only current meta-device model layout, never actors or native worlds."""
    from network.rlm import ATCRLM
    from training.ppo_config import POLICY_ID

    spec = policy_module_spec(run)
    with torch.device("meta"):
        module = spec.build()
    if (
        type(module) is not ATCRLM
        or any(value.device.type != "meta" for value in module.parameters())
        or any(value.device.type not in {"meta", "cpu"} for value in module.buffers())
    ):
        raise CheckpointBoundaryError("unsupported current meta model layout")
    _tensor_layout(state["learner_group"]["learner"]["rl_module"], {POLICY_ID: module.state_dict()})
