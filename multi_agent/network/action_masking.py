"""Action-mask aware RLModule base: strips the mask and applies it to logits."""

from typing import Any, cast

import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN

from actions.actions import Action


class ActionMaskingRLModule(TorchRLModule):
    """A TorchRLModule that implements an action masking for safe RL.

    This RLModule implements action masking to avoid unsafe/unwanted actions
    dependent on the current state (observations). It does so by using an
    environment generated action mask defining which actions are allowed and
    which should be avoided. The action mask is extracted from the
    environment's `gymnasium.spaces.dict.Dict` observation and applied after
    the module's `forward`-pass to the action logits. The resulting action
    logits prevent unsafe/unwanted actions to be sampled from the corresponding
    action distribution.

    Note, this RLModule is implemented for the `PPO` algorithm only. It is not
    guaranteed to work with other algorithms. Furthermore, not that for this
    module to work it requires an environment with a `gymnasium.spaces.dict.Dict`
    observation space containing two keys, `"action_mask"` and `"observations"`.
    """

    @override(RLModule)
    def __init__(
        self,
        *,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        inference_only: bool | None = None,
        learner_only: bool = False,
        model_config: dict | DefaultModelConfig | None = None,
        catalog_class=None,
        **kwargs,
    ):
        # If observation space is not of type `Dict` raise an error.
        if not isinstance(observation_space, gym.spaces.dict.Dict):
            raise ValueError(
                "This RLModule requires the environment to provide a "
                "`gym.spaces.Dict` observation space of the form: \n"
                " {'action_mask': <mask_space>, 'observations': <obs_space>}"
            )

        # While the environment holds an observation space that contains, both,
        # the action mask and the original observation space, the 'RLModule'
        # receives only the `"observation"` element of the space, but not the
        # action mask.
        self.observation_space_with_mask = observation_space
        self.observation_space = observation_space["observations"]

        # The ATCRLM, in its constructor will build networks for the
        # original observation space (i.e. without the action mask).
        super().__init__(
            observation_space=self.observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    def _preprocess_batch(self, batch: dict[str, Any], **_) -> tuple[torch.Tensor, dict[str, Any]]:
        # Extract the available actions tensor from the observation without mutating
        # the original batch (which may be reused by RLlib in other places).
        obs_with_mask = cast(dict[str, Any], batch[Columns.OBS])
        action_mask = obs_with_mask["action_mask"]

        # Create a shallow copy of the batch and replace the OBS with the inner
        # "observations" dict. This keeps the original batch intact.
        new_batch = dict(batch)
        inner_obs = cast(dict[str, torch.Tensor], obs_with_mask["observations"])
        new_batch[Columns.OBS] = dict(inner_obs)

        return action_mask, new_batch

    def _apply_action_mask(
        self, action_logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply action mask by setting disallowed logits to -inf."""
        m = self._align_action_tensor(
            action_mask, action_logits=action_logits, tensor_name="action_mask"
        )
        # Keep at least one action valid when a mask row is entirely zeros.
        flat_m = m.reshape(-1, m.shape[-1])
        all_zero = (flat_m <= 0).all(dim=-1)
        if all_zero.any():
            flat_m = flat_m.clone()
            flat_m[all_zero, int(Action.DO_NOTHING)] = 1.0
            m = flat_m.reshape_as(m)
        return action_logits.masked_fill(m <= 0, FLOAT_MIN)

    @staticmethod
    def _align_action_tensor(
        tensor: torch.Tensor,
        *,
        action_logits: torch.Tensor,
        tensor_name: str,
    ) -> torch.Tensor:
        """Broadcast an action-wise tensor to match action logits shape."""
        if action_logits.dim() not in (2, 3):
            raise ValueError(
                f"Action logits must be rank 2 or 3, got rank {action_logits.dim()} "
                f"with shape {tuple(action_logits.shape)}"
            )

        aligned = tensor.to(device=action_logits.device, dtype=action_logits.dtype)
        logits_3d = action_logits if action_logits.dim() == 3 else action_logits.unsqueeze(1)
        batch_size, seq_len, num_actions = logits_3d.shape

        if aligned.dim() == 1:
            aligned = aligned.unsqueeze(0).unsqueeze(0)  # [A] -> [1, 1, A]
        elif aligned.dim() == 2:
            aligned = aligned.unsqueeze(1)  # [B or 1, A] -> [B or 1, 1, A]
        elif aligned.dim() != 3:
            raise ValueError(
                f"{tensor_name} must be rank 1, 2, or 3; got rank {aligned.dim()} "
                f"with shape {tuple(aligned.shape)}"
            )

        if aligned.shape[-1] != num_actions:
            raise ValueError(
                f"{tensor_name} last dimension {aligned.shape[-1]} does not match "
                f"action logits {num_actions}"
            )

        if aligned.shape[0] not in (1, batch_size):
            raise ValueError(
                f"{tensor_name} batch dimension {aligned.shape[0]} is incompatible with "
                f"logits batch size {batch_size}"
            )
        if aligned.shape[1] not in (1, seq_len):
            raise ValueError(
                f"{tensor_name} time dimension {aligned.shape[1]} is incompatible with "
                f"logits sequence length {seq_len}"
            )

        expanded = aligned.expand(batch_size, seq_len, num_actions)
        return expanded if action_logits.dim() == 3 else expanded[:, 0, :]
