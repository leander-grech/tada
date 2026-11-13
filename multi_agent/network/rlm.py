"""RLModule for the ATC environment.

Architecture diagram: docs/architecture.mmd
"""

from typing import Any, cast

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.configs import MLPHeadConfig
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

from config.settings import ArchitectureSettings
from network.action_masking import ActionMaskingRLModule
from network.encoders import (
    CrossAttentionNeighborEncoder,
    FlightPlanConvEncoder,
)
from network.recurrence import RecurrenceMixin
from network.value_normalizer import RunningMeanStd

# Bump on any change that alters outputs; checkpoints record this token and reject mismatches.
MODEL_VERSION = "ma-model-v1"


class ATCRLM(RecurrenceMixin, ActionMaskingRLModule, ValueFunctionAPI):  # pylint: disable=too-many-ancestors
    """Custom encoder using RLModule API for Dict observation space
    with separate encoders for aircraft scalars, flight plans, and neighbors."""

    def __init__(self, **kwargs):
        # Encoders initialized in setup() after super().__init__
        self.aircraft_scalar_encoder: nn.Module | None = None
        self.flight_plan_encoder: FlightPlanConvEncoder | None = None
        self.combined_encoder: nn.Module | None = None
        self.neighbor_encoder: nn.Module | None = None
        self.aircraft_norm: nn.LayerNorm | None = None
        self.flight_plan_norm: nn.LayerNorm | None = None
        self.combined_norm: nn.LayerNorm | None = None
        self.post_combined_norm: nn.LayerNorm | None = None
        self.combined_skip: nn.Linear | None = None
        self.combined_skip_gate: nn.Parameter | None = None
        self.critic_context_encoder: nn.Module | None = None
        self.critic_context_norm: nn.LayerNorm | None = None
        self.gru: nn.GRU | None = None
        self.gru_norm: nn.LayerNorm | None = None
        self.value_normalizer: RunningMeanStd | None = None
        self._skip_value_denorm: bool = False

        super().__init__(**kwargs)

    @override(TorchRLModule)
    def set_state(self, state):
        # Ray's conversion unconditionally narrows floating buffers to float32.
        # Validate the complete destination layout before loading, preserving
        # native tensor dtypes and interpreting exported scalars by target dtype.
        actual = self.state_dict()
        if set(actual) - set(state) or (not self.inference_only and set(state) - set(actual)):
            raise ValueError("missing or unknown module checkpoint parameters")
        prepared = {}
        for name, target in actual.items():
            value = state[name]
            if type(value) in (int, float) and target.ndim == 0:
                tensor = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            else:
                # Ray object-store arrays are read-only; own the temporary
                # tensor so native load cannot alias that immutable storage.
                tensor = (
                    torch.tensor(value, device=target.device)
                    if not isinstance(value, torch.Tensor)
                    else value.to(target.device)
                )
            if (
                tensor.shape != target.shape
                or tensor.dtype != target.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("module checkpoint shape/dtype/state mismatch")
            prepared[name] = tensor
        self.load_state_dict(prepared, strict=True)

    @override(TorchRLModule)
    def setup(self):
        """Set up the neural network architecture."""
        super().setup()
        # We need to reset here the observation space such that the
        # super`s (`TorchRLModule`) observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

        from common.hashing import FrozenDict
        from common.sealed import consume
        from models.observation_settings import ObservationSettings
        from models.observation_space import build_observation_space

        self.consumed_semantic_digest = None
        payload = self.model_config.get("resolved_model")
        if payload is not None:
            self.consumed_semantic_digest = consume(
                payload, self.model_config.get("expected_semantic_digest"), role="model"
            )
            settings = ObservationSettings.from_features(payload["features"])
            if self.observation_space != build_observation_space(settings):
                raise ValueError("model observation space differs from bound feature dimensions")
            self.architecture = FrozenDict(payload["features"]["architecture"])
        else:
            self.architecture = FrozenDict(self.model_config["architecture"])
        self.model_config["max_seq_len"] = (
            payload["features"]["architecture"]["max_seq_len"]
            if payload is not None
            else self.model_config.get("max_seq_len", self.architecture["max_seq_len"])
        )

        self._create_encoders()

        hidden = int(self.architecture["mlp_hidden"])
        encoder_output_dim = int(self.architecture["gru_hidden"])
        critic_input_dim = encoder_output_dim + int(self.architecture["critic_context_out"])
        assert self.action_space is not None, "Action space must be set"
        action_space_n = getattr(self.action_space, "n", None)
        assert action_space_n is not None, "Action space must be discrete with 'n' attribute"
        pi_config = MLPHeadConfig(
            input_dims=[encoder_output_dim],
            hidden_layer_dims=[2 * hidden, hidden],
            hidden_layer_activation="silu",
            output_layer_dim=action_space_n,
            output_layer_weights_initializer=nn.init.orthogonal_,
            output_layer_weights_initializer_config={"gain": 0.01},
            output_layer_bias_initializer=nn.init.zeros_,
        )
        vf_config = MLPHeadConfig(
            input_dims=[critic_input_dim],
            hidden_layer_dims=[2 * hidden, hidden],
            hidden_layer_activation="silu",
            output_layer_dim=1,
            hidden_layer_use_layernorm=True,
            output_layer_weights_initializer=nn.init.orthogonal_,
            output_layer_weights_initializer_config={"gain": 0.5},
            output_layer_bias_initializer=nn.init.zeros_,
        )

        # Policy head (actor)
        self.policy_head = pi_config.build(framework="torch")
        # Value head (critic)
        self.value_head = vf_config.build(framework="torch")
        # Value normalizer (MAPPO-style, updated by the learner during training)
        self.value_normalizer = RunningMeanStd()

    @override(RLModule)
    def get_initial_state(self) -> dict:
        """Return initial GRU hidden state for RLlib's native stateful handling."""
        return {
            "gru_h": torch.zeros(int(self.architecture["gru_hidden"]), dtype=torch.float32),
        }

    def _create_encoders(self) -> None:
        # Use config-defined sizes
        aircraft_scalar_len = int(
            self.observation_space["observations"]["aircraft_scalar"].shape[0]
        )
        waypoint_dim = int(self.observation_space["observations"]["flight_plans_rel"].shape[1])
        max_waypoints = int(self.observation_space["observations"]["flight_plans_rel"].shape[0])

        # Aircraft scalar encoder: scalar_len → hidden → out.
        aircraft_scalar_hidden = int(self.architecture["aircraft_scalar_hidden"])
        aircraft_scalar_out = int(self.architecture["aircraft_scalar_out"])
        self.aircraft_scalar_encoder = nn.Sequential(
            nn.Linear(aircraft_scalar_len, aircraft_scalar_hidden),
            nn.SiLU(),
            nn.Linear(aircraft_scalar_hidden, aircraft_scalar_out),
            nn.SiLU(),
        )

        # Flight plan encoder using 1D CNN with masked average pooling
        flight_plan_hidden = int(self.architecture["flight_plan_hidden"])
        conv1_channels = self.architecture["conv1_channels"]
        conv2_channels = self.architecture["conv2_channels"]
        self.flight_plan_encoder = FlightPlanConvEncoder(
            waypoint_dim=waypoint_dim,
            hidden_dim=flight_plan_hidden,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            max_waypoints=max_waypoints,
        )

        # Calculate flight plan encoder output size with dummy input
        with torch.no_grad():
            dummy_flight_plan = torch.zeros(1, max_waypoints, waypoint_dim)
            flight_plan_out = self.flight_plan_encoder(dummy_flight_plan).shape[1]

        self.aircraft_norm = nn.LayerNorm(aircraft_scalar_out)
        self.flight_plan_norm = nn.LayerNorm(flight_plan_out)

        # Neighbor encoder: ego-query cross-attention over neighbor features.
        ego_dim = aircraft_scalar_out + flight_plan_out
        self.neighbor_encoder = CrossAttentionNeighborEncoder(
            ego_dim=ego_dim,
            neighbor_feature_dim=int(
                self.observation_space["observations"]["neighbor_features"].shape[1]
            ),
            model_dim=int(self.architecture["neighbor_model"]),
            num_heads=int(self.architecture["neighbor_heads"]),
        )
        neighbor_out_dim = int(self.architecture["neighbor_model"])

        total_features_in = ego_dim + neighbor_out_dim
        total_features_hidden = int(self.architecture["combined_hidden"])
        total_features_out = int(self.architecture["combined_out"])

        self.combined_encoder = nn.Sequential(
            nn.Linear(in_features=total_features_in, out_features=total_features_hidden),
            nn.SiLU(),
            nn.Linear(in_features=total_features_hidden, out_features=total_features_out),
        )

        # Skip connection: project input directly to combined output dimension
        self.combined_skip = nn.Linear(total_features_in, total_features_out)
        # Learnable scalar gate: sigmoid(0.0) = 0.5 → starts as balanced 50/50 blend
        # of MLP and linear paths, then adjusts during training.
        self.combined_skip_gate = nn.Parameter(torch.tensor(0.0))

        self.combined_norm = nn.LayerNorm(total_features_in)
        self.post_combined_norm = nn.LayerNorm(total_features_out)

        # GRU temporal memory (input = combined encoder output directly)
        self.gru = nn.GRU(
            input_size=total_features_out,
            hidden_size=int(self.architecture["gru_hidden"]),
            num_layers=1,
            batch_first=True,
        )
        self.gru_norm = nn.LayerNorm(int(self.architecture["gru_hidden"]))

        critic_context_out = int(self.architecture["critic_context_out"])
        self.critic_context_encoder = nn.Sequential(
            nn.Linear(
                int(self.observation_space["observations"]["global_context"].shape[0]),
                critic_context_out,
            ),
            nn.SiLU(),
        )
        self.critic_context_norm = nn.LayerNorm(critic_context_out)

    def _assert_encoders_initialized(self) -> None:
        missing = [
            name
            for name, value in (
                ("aircraft_scalar_encoder", self.aircraft_scalar_encoder),
                ("flight_plan_encoder", self.flight_plan_encoder),
                ("neighbor_encoder", self.neighbor_encoder),
                ("combined_encoder", self.combined_encoder),
                ("combined_skip", self.combined_skip),
                ("combined_skip_gate", self.combined_skip_gate),
                ("aircraft_norm", self.aircraft_norm),
                ("flight_plan_norm", self.flight_plan_norm),
                ("combined_norm", self.combined_norm),
                ("post_combined_norm", self.post_combined_norm),
                ("gru", self.gru),
                ("gru_norm", self.gru_norm),
                ("critic_context_encoder", self.critic_context_encoder),
                ("critic_context_norm", self.critic_context_norm),
            )
            if value is None
        ]
        assert not missing, f"Encoders not initialized: {', '.join(missing)}"

    def _encode_observations(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode the dictionary observations into a single feature vector.

        Handles both flat batches [B, D] and time-series batches [B, T, D] from
        stateful module processing.
        """
        self._assert_encoders_initialized()
        aircraft_scalar_encoder = cast(nn.Module, self.aircraft_scalar_encoder)
        flight_plan_encoder = cast(FlightPlanConvEncoder, self.flight_plan_encoder)
        neighbor_encoder = cast(nn.Module, self.neighbor_encoder)
        combined_encoder = cast(nn.Module, self.combined_encoder)
        aircraft_norm = cast(nn.LayerNorm, self.aircraft_norm)
        flight_plan_norm = cast(nn.LayerNorm, self.flight_plan_norm)
        combined_norm = cast(nn.LayerNorm, self.combined_norm)

        # Check for time dimension by looking at aircraft_scalar shape.
        aircraft_scalar_raw = obs_dict["aircraft_scalar"]
        has_time_dim = aircraft_scalar_raw.dim() == 3
        batch_size = aircraft_scalar_raw.shape[0]
        seq_len = aircraft_scalar_raw.shape[1] if has_time_dim else 1

        # Flatten time dimension for all observation components
        aircraft_scalar = self._flatten_time(aircraft_scalar_raw, batch_size, seq_len, has_time_dim)
        flight_plans = self._flatten_time(
            obs_dict["flight_plans_rel"], batch_size, seq_len, has_time_dim
        )
        neighbor_feats = self._flatten_time(
            obs_dict["neighbor_features"], batch_size, seq_len, has_time_dim
        )
        neighbor_mask = self._flatten_time(
            obs_dict["neighbor_mask"], batch_size, seq_len, has_time_dim
        )
        flight_plan_mask = obs_dict.get("flight_plan_mask")
        if flight_plan_mask is not None:
            flight_plan_mask = self._flatten_time(
                flight_plan_mask, batch_size, seq_len, has_time_dim
            )

        aircraft_encoded = aircraft_norm(aircraft_scalar_encoder(aircraft_scalar))
        flight_plans_encoded = flight_plan_norm(
            flight_plan_encoder(
                flight_plans,
                flight_plan_mask.float() if flight_plan_mask is not None else None,
            )
        )

        ego_embed = torch.cat([aircraft_encoded, flight_plans_encoded], dim=1)

        neighbors_encoded = neighbor_encoder(ego_embed, neighbor_feats, neighbor_mask)
        combined_features = combined_norm(torch.cat([ego_embed, neighbors_encoded], dim=1))

        combined_skip = cast(nn.Linear, self.combined_skip)
        combined_skip_gate = cast(nn.Parameter, self.combined_skip_gate)
        post_combined_norm = cast(nn.LayerNorm, self.post_combined_norm)
        skip_scale = torch.sigmoid(combined_skip_gate)
        encoded = post_combined_norm(
            combined_encoder(combined_features) + skip_scale * combined_skip(combined_features)
        )
        # Restore time dimension: [B*T, D] -> [B, T, D]
        return self._unflatten_time(encoded, batch_size, seq_len, has_time_dim)

    def _encode_critic_context(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode critic-only global context without affecting the actor path."""
        critic_context_encoder = cast(nn.Module, self.critic_context_encoder)
        critic_context_norm = cast(nn.LayerNorm, self.critic_context_norm)

        global_context_raw = obs_dict["global_context"]
        has_time_dim = global_context_raw.dim() == 3
        batch_size = global_context_raw.shape[0]
        seq_len = global_context_raw.shape[1] if has_time_dim else 1
        global_context = self._flatten_time(global_context_raw, batch_size, seq_len, has_time_dim)
        encoded = critic_context_norm(critic_context_encoder(global_context))
        return self._unflatten_time(encoded, batch_size, seq_len, has_time_dim)

    def _encode_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """Encode preprocessed batch observations and apply GRU temporal memory."""
        embeddings = self._encode_observations(cast(dict[str, torch.Tensor], batch[Columns.OBS]))
        embeddings, state_out = self._apply_gru(embeddings, batch)
        return embeddings, state_out

    @staticmethod
    def _add_state_out(result: dict[str, Any], state_out: dict) -> dict[str, Any]:
        """Attach recurrent state to model outputs."""
        result[Columns.STATE_OUT] = state_out
        return result

    def _forward_common(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Shared forward logic for inference, exploration, and training."""
        action_mask, batch = self._preprocess_batch(batch)
        embeddings, state_out = self._encode_batch(batch)
        action_logits: torch.Tensor = self.policy_head(embeddings)
        action_dist_inputs = self._apply_action_mask(action_logits, action_mask)
        return action_dist_inputs, embeddings, state_out

    @override(TorchRLModule)
    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        with torch.no_grad():
            action_dist_inputs, _embeddings, state_out = self._forward_common(batch)
            result: dict[str, Any] = {Columns.ACTION_DIST_INPUTS: action_dist_inputs}
            return self._add_state_out(result, state_out)

    @override(TorchRLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        action_dist_inputs, embeddings, state_out = self._forward_common(batch)
        result: dict[str, Any] = {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.EMBEDDINGS: embeddings,
        }
        return self._add_state_out(result, state_out)

    @override(TorchRLModule)
    def _forward_exploration(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Forward pass for exploration (action sampling without gradients)."""
        with torch.no_grad():
            return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: dict[str, Any], embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        _action_mask, batch = self._preprocess_batch(batch)
        obs_dict = cast(dict[str, torch.Tensor], batch[Columns.OBS])
        if embeddings is None:
            embeddings, _state_out = self._encode_batch(batch)
        critic_context = self._encode_critic_context(obs_dict)
        if embeddings.dim() == 3 and critic_context.dim() == 2:
            critic_context = critic_context.unsqueeze(1).expand(-1, embeddings.shape[1], -1)
        elif embeddings.dim() == 2 and critic_context.dim() == 3:
            critic_context = critic_context[:, -1, :]
        value_inputs = torch.cat([embeddings, critic_context], dim=-1)
        values: torch.Tensor = self.value_head(value_inputs).squeeze(-1)

        # During training loss computation, the learner sets _skip_value_denorm=True
        # so the loss operates in normalized space (matching normalized targets).
        if self.value_normalizer is not None and not self._skip_value_denorm:
            values = self.value_normalizer.denormalize(values)

        return values


def unbound_model_config(architecture: ArchitectureSettings) -> dict:
    """Model config for modules built without a bound training contract."""
    return {
        "architecture": {
            "mlp_hidden": architecture.mlp_hidden,
            "gru_hidden": architecture.gru_hidden_size,
            "critic_context_out": architecture.critic_context_out,
            "aircraft_scalar_hidden": architecture.aircraft_scalar_hidden,
            "aircraft_scalar_out": architecture.aircraft_scalar_out,
            "flight_plan_hidden": architecture.flight_plan_hidden,
            "conv1_channels": architecture.flight_plan_conv1_channels,
            "conv2_channels": architecture.flight_plan_conv2_channels,
            "neighbor_model": architecture.nei_d_model,
            "neighbor_heads": architecture.nei_attn_heads,
            "combined_hidden": architecture.combined_encoder_hidden,
            "combined_out": architecture.combined_encoder_out,
            "max_seq_len": architecture.gru_max_seq_len,
        }
    }
