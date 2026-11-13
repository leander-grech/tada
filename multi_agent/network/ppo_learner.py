"""Custom PPO learner with MAPPO-style value normalization."""

from copy import deepcopy
from math import isfinite
from typing import Any, cast

import torch
import tree
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import ALL_MODULES
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ModuleID

from network.learner_connectors import install_gae_connector
from network.rlm import ATCRLM


class ValueNormalizedPPOLearner(PPOTorchLearner):
    """PPO learner with MAPPO-style value normalization.

    Normalizes both value predictions and value targets by running mean/std
    before the VF loss, making the critic loss scale-invariant across
    scenario sizes.
    """

    @override(PPOTorchLearner)
    def build(self) -> None:
        if self._is_built:
            return
        super().build()
        self._checkpoint_learned = {}
        self._checkpoint_updates = 0
        self._optimizer_work = {}
        self._checkpoint_optimizer_work = {}
        self._completed_optimizer_work = {}
        config = cast(PPOConfig, self.config)
        install_gae_connector(self._learner_connector, gamma=config.gamma, lambda_=config.lambda_)

    def _effective_state(self) -> dict:
        modules = {}
        for module_id in self.module.keys():
            config = cast(PPOConfig, self.config.get_config_for_module(module_id))
            if (
                config.use_kl_loss
                or isinstance(config.lr, (list, tuple))
                or isinstance(config.entropy_coeff, (list, tuple))
            ):
                raise ValueError(
                    "unsupported checkpoint contract: constant LR/entropy and inactive KL required"
                )
            optimizers = self._module_optimizers[module_id]
            if len(optimizers) != 1:
                raise ValueError("unsupported effective optimizer layout")
            optimizer = self._named_optimizers[optimizers[0]]
            scheduler = self._optimizer_lr_schedules[optimizer]
            entropy = self.entropy_coeff_schedulers_per_module[module_id]
            if scheduler.use_schedule or entropy.use_schedule:
                raise ValueError("unsupported checkpoint contract: nonconstant scheduler")
            lr, coefficient = scheduler.get_current_value(), entropy.get_current_value()
            if any(group["lr"] != lr for group in optimizer.param_groups):
                raise ValueError("effective learning rate differs from scheduler")
            modules[module_id] = {"learning_rate": float(lr), "entropy_coeff": float(coefficient)}
        return {
            "version": 2,
            "modules": modules,
            "learned": deepcopy(self._checkpoint_learned),
            "updates": self._checkpoint_updates,
            "optimizer_work": deepcopy(self._checkpoint_optimizer_work),
        }

    @staticmethod
    def _validate_overrides(overrides, *, complete=False):
        keys = {"learning_rate", "entropy_coeff"}
        if (
            not isinstance(overrides, dict)
            or set(overrides) - keys
            or complete
            and set(overrides) != keys
        ):
            raise ValueError("unknown effective learner sidecar or phase override")
        for key, value in overrides.items():
            if (
                type(value) not in (float, int)
                or not isfinite(value)
                or value < 0
                or key == "learning_rate"
                and value == 0
            ):
                raise ValueError("phase overrides require finite positive LR/nonnegative entropy")

    def _apply_constants(self, module_id, values):
        config = cast(PPOConfig, self.config.get_config_for_module(module_id))
        for name in self._module_optimizers[module_id]:
            optimizer = self._named_optimizers[name]
            self._optimizer_lr_schedules[optimizer] = Scheduler(
                values["learning_rate"], framework=self.framework, device=self._device
            )
            self._set_optimizer_lr(optimizer, lr=values["learning_rate"])
        self.entropy_coeff_schedulers_per_module[module_id] = Scheduler(
            values["entropy_coeff"], framework=self.framework, device=self._device
        )
        config.lr = values["learning_rate"]
        config.entropy_coeff = values["entropy_coeff"]

    def apply_phase_overrides(self, overrides: dict[str, float]) -> dict[str, float]:
        """Apply only learning_rate/entropy_coeff, without touching learned state."""
        self._validate_overrides(overrides)
        current = self._effective_state()["modules"]
        if len(current) != 1:
            raise ValueError("phase overrides require one shared module")
        module_id, values = next(iter(current.items()))
        effective = dict(values, **overrides)
        self._apply_constants(module_id, effective)
        return effective

    @staticmethod
    def _iter_policy_batches(batch: Any, module_ids: list[ModuleID]) -> list[tuple[ModuleID, Any]]:
        policy_batches = getattr(batch, "policy_batches", None)
        if isinstance(policy_batches, dict):
            return [(module_id, module_batch) for module_id, module_batch in policy_batches.items()]
        if len(module_ids) == 1 and batch is not None:
            return [(module_ids[0], batch)]
        return []

    def _update_value_normalizers_from_batches(self, batches: list[Any]) -> None:
        """Update value normalizers once from full train batches before SGD slicing."""
        from ray.rllib.evaluation.postprocessing import Postprocessing

        module_ids = list(self.module.keys())
        targets_by_module: dict[ModuleID, list[torch.Tensor]] = {}
        for batch in batches:
            for module_id, module_batch in self._iter_policy_batches(batch, module_ids):
                value_targets = module_batch.get(Postprocessing.VALUE_TARGETS)
                if value_targets is None:
                    continue
                module: ATCRLM = self.module[module_id].unwrapped()  # ty: ignore[invalid-assignment]
                normalizer = module.value_normalizer
                if normalizer is None:
                    continue
                tensor = torch.as_tensor(
                    value_targets,
                    dtype=torch.float32,
                    device=normalizer.mean_acc.device,
                )
                loss_mask = module_batch.get(Columns.LOSS_MASK)
                if loss_mask is None:
                    # Only flat, nonrecurrent caller-supplied batches are unpadded.
                    # Episode connectors always supply a mask, including bootstrap rows.
                    if module.is_stateful() or tensor.ndim != 1:
                        raise ValueError("value statistics require a loss mask for padded targets")
                    selected = tensor
                else:
                    mask = torch.as_tensor(
                        loss_mask, dtype=torch.bool, device=normalizer.mean_acc.device
                    )
                    if mask.shape != tensor.shape:
                        raise ValueError("value target and loss mask shape must match")
                    selected = tensor.reshape(-1)[mask.reshape(-1)]
                if selected.numel() == 0:
                    continue
                targets_by_module.setdefault(module_id, []).append(selected)

        for module_id, tensors in targets_by_module.items():
            if not tensors:
                continue
            module: ATCRLM = self.module[module_id].unwrapped()  # ty: ignore[invalid-assignment]
            normalizer = module.value_normalizer
            if normalizer is not None:
                normalizer.update(torch.cat(tensors, dim=0))

    @override(PPOTorchLearner)
    def _make_batch_if_necessary(self, training_data: Any) -> Any:
        batch = super()._make_batch_if_necessary(training_data)
        # GAE has already consumed all N+1 values. Only now can a sequence
        # containing solely its artificial final observation/padding be removed.
        for module_id, module_batch in batch.policy_batches.items():
            if not self.module[module_id].is_stateful():
                continue
            mask = module_batch[Columns.LOSS_MASK]
            keep = mask.any(dim=1)
            if not keep.any():
                raise ValueError("recurrent learner batch has no valid transitions")
            if keep.all():
                continue
            # Every connector column (including nested state_in) shares B.
            # Reconstruct to refresh count=sum(seq_lens); keep T and recurrence.
            selected = tree.map_structure(lambda value: value[keep], dict(module_batch))
            batch.policy_batches[module_id] = SampleBatch(
                selected,
                _zero_padded=module_batch.zero_padded,
                _max_seq_len=module_batch.max_seq_len,
                _time_major=module_batch.time_major,
                _is_training=module_batch.is_training,
                _num_grad_updates=module_batch.num_grad_updates,
            )
        self._update_value_normalizers_from_batches([batch])
        return batch

    def _log_steps_trained_metrics(self, batch):
        # Ray invokes this only after _update has completed the optimizer step.
        # Retain its stock metrics, which include artificial rows and repeat
        # full-batch env counts. Measure valid work separately, including wraps.
        super()._log_steps_trained_metrics(batch)
        valid_rows = 0
        for module_id, module_batch in batch.policy_batches.items():
            mask = module_batch.get(Columns.LOSS_MASK)
            rows = int(mask.sum()) if mask is not None else len(module_batch)
            valid_rows += rows
            work = self._optimizer_work.setdefault(module_id, [0, 0])
            work[0] += rows
            work[1] += 1
        work = self._optimizer_work.setdefault(ALL_MODULES, [0, 0])
        work[0] += valid_rows
        work[1] += 1

    def after_gradient_based_update(self, *, timesteps):
        super().after_gradient_based_update(timesteps=timesteps)
        # These are this completed update's observations, not durable lifetime
        # totals. Failed attempts publish no completed-update work or receipt.
        work, self._optimizer_work = self._optimizer_work, {}
        if not work:
            return
        self._completed_optimizer_work = deepcopy(work)
        # Ray retains empty reduced Stats for absent modules. Emit truthful zero
        # observations so a later p-only update cannot return stale/NaN q work.
        for module_id in [*self.module.keys(), ALL_MODULES]:
            rows, minibatches = work.get(module_id, (0, 0))
            for name, value in (
                ("actual_valid_optimizer_rows_this_update", rows),
                ("actual_optimizer_minibatches_this_update", minibatches),
                ("actual_optimizer_updates_this_update", int(minibatches > 0)),
            ):
                self.metrics.log_value((module_id, name), value, reduce="sum")

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: "PPOConfig",
        batch: dict[str, Any],
        fwd_out: dict[str, TensorType],
    ) -> TensorType:
        from ray.rllib.evaluation.postprocessing import Postprocessing

        module: ATCRLM = self.module[module_id].unwrapped()  # ty: ignore[invalid-assignment]
        normalizer = module.value_normalizer

        # Normalize value targets and tell the model to skip denormalization
        # so the PPO loss operates entirely in normalized space.
        value_targets = batch[Postprocessing.VALUE_TARGETS]
        batch[Postprocessing.VALUE_TARGETS] = normalizer.normalize(value_targets)

        module._skip_value_denorm = True
        try:
            loss = super().compute_loss_for_module(
                module_id=module_id,
                config=config,
                batch=batch,
                fwd_out=fwd_out,
            )
        finally:
            module._skip_value_denorm = False

        return loss
