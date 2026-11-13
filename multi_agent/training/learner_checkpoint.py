"""Checkpointable PPO learner: effective-state sidecar, optimizer slots and receipts."""

from copy import deepcopy
from typing import Any, cast

from ray.rllib.core import ALL_MODULES

from network.ppo_learner import ValueNormalizedPPOLearner
from training.checkpoint_boundary import advance_learned_receipts, validate_receipts


class LearnerCheckpointMixin:
    """Save, validate and restore the learner's exact effective training state."""

    def get_state(self: "ATCPPOLearner", components=None, *, not_components=None, **kwargs):
        state = super().get_state(components, not_components=not_components, **kwargs)
        if "optimizer" in state:
            state["atc_effective_state"] = self._effective_state()
        return state

    def validate_checkpoint_state(self: "ATCPPOLearner", state):
        """Preflight effective constants and actual optimizer slots without mutation."""
        effective = state.get("atc_effective_state")
        if "optimizer" in state or effective is not None:
            if (
                not isinstance(effective, dict)
                or set(effective) != {"version", "modules", "learned", "updates", "optimizer_work"}
                or effective["version"] != 2
            ):
                raise ValueError("missing or unknown effective learner sidecar")
            self._effective_state()  # Reject unsupported destination schedules/KL before mutation.
            if set(effective["modules"]) != set(self.module.keys()):
                raise ValueError("effective learner sidecar module mismatch")
            for values in effective["modules"].values():
                self._validate_overrides(values, complete=True)
            from network.optimizer_state import restore_adam

            if "optimizer" not in state or set(state["optimizer"]) != set(self._named_optimizers):
                raise ValueError("incomplete effective optimizer sidecar")
            for name, envelope in state["optimizer"].items():
                if (
                    set(envelope) != {"module_id", "state"}
                    or envelope["module_id"] != self._optimizer_name_to_module[name]
                ):
                    raise ValueError("optimizer sidecar module mismatch")
                restore_adam(
                    deepcopy(self._named_optimizers[name]), envelope["state"], self._device
                )
                expected_lr = effective["modules"][envelope["module_id"]]["learning_rate"]
                if any(group["lr"] != expected_lr for group in envelope["state"]["param_groups"]):
                    raise ValueError("optimizer and effective sidecar LR mismatch")
            validate_receipts(effective["learned"])
            work = effective["optimizer_work"]
            if not isinstance(work, dict) or set(work) - (set(self.module.keys()) | {ALL_MODULES}):
                raise ValueError("invalid lifetime optimizer work modules")
            for counts in work.values():
                if (
                    not isinstance(counts, dict)
                    or set(counts) != {"valid_rows", "minibatches", "updates"}
                    or any(type(value) is not int or value < 0 for value in counts.values())
                ):
                    raise ValueError("invalid lifetime optimizer work counters")
            if type(effective["updates"]) is not int or effective["updates"] < 0:
                raise ValueError("invalid learner sidecar update count")
        return effective

    def set_state(self: "ATCPPOLearner", state):
        effective = self.validate_checkpoint_state(state)
        super().set_state(state)
        if effective is not None:
            for module_id, values in effective["modules"].items():
                self._apply_constants(module_id, values)
            self._checkpoint_learned = deepcopy(effective["learned"])
            self._checkpoint_updates = effective["updates"]
            self._checkpoint_optimizer_work = deepcopy(effective["optimizer_work"])

    def _set_optimizer_state(self: "ATCPPOLearner", state):
        from network.optimizer_state import restore_adam

        for name, envelope in state.items():
            if set(envelope) != {"module_id", "state"} or name not in self._named_optimizers:
                raise ValueError("unsupported optimizer checkpoint envelope")
            restore_adam(self._named_optimizers[name], envelope["state"], self._device)

    def update(self: "ATCPPOLearner", *args, **kwargs):
        # Ray chooses int(B * minibatch_size / sum(seq_lens)) sequences.
        # Requiring at least T makes that positive for every supported BxT batch,
        # before parent update mutates schedulers, weight sequence or statistics.
        minibatch_size = kwargs.get("minibatch_size")
        if minibatch_size is not None:
            for module_id in self.module.keys():
                module = self.module[module_id]
                model_config = cast(dict[str, Any], module.model_config)
                if module.is_stateful() and minibatch_size < model_config["max_seq_len"]:
                    raise ValueError("recurrent minibatch_size must be at least max_seq_len")
        data = kwargs.get("training_data")
        if (
            kwargs.get("episodes_refs") is not None
            or data is not None
            and data.episodes_refs is not None
        ):
            raise ValueError("checkpoint receipts require materialized episodes, not lossy refs")
        episodes = data.episodes if data is not None else kwargs.get("episodes")
        learned = advance_learned_receipts(self._checkpoint_learned, episodes or [])
        self._optimizer_work = {}
        self._completed_optimizer_work = {}
        try:
            result = super().update(*args, **kwargs)
        finally:
            self._optimizer_work = {}
        for module_id, (rows, minibatches) in self._completed_optimizer_work.items():
            counts = self._checkpoint_optimizer_work.setdefault(
                module_id, {"valid_rows": 0, "minibatches": 0, "updates": 0}
            )
            counts["valid_rows"] += rows
            counts["minibatches"] += minibatches
            counts["updates"] += int(minibatches > 0)
        self._completed_optimizer_work = {}
        self._checkpoint_learned = learned
        self._checkpoint_updates += 1
        return result


class ATCPPOLearner(LearnerCheckpointMixin, ValueNormalizedPPOLearner):
    """The owned PPO learner: value-normalized loss plus checkpoint machinery."""
