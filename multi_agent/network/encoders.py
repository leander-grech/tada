"""Flight-plan and neighbor encoders of the ATC policy network."""

import torch
import torch.nn.functional as F
from torch import nn


def append_null_neighbor_feature(
    neighbor_features: torch.Tensor,
    neighbor_mask: torch.Tensor,
    null_neighbor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append a learned null neighbor and activate it only for empty rows."""
    batch_size = neighbor_features.shape[0]
    no_neighbors = (neighbor_mask > 0).sum(dim=1, keepdim=True) == 0
    null_feat = null_neighbor.expand(batch_size, 1, -1)
    features = torch.cat([neighbor_features, null_feat], dim=1)
    mask = torch.cat([neighbor_mask, no_neighbors.to(dtype=neighbor_mask.dtype)], dim=1)
    return features, mask


class FlightPlanConvEncoder(nn.Module):
    """Lightweight 1D CNN encoder with masked average pooling for flight plans."""

    def __init__(
        self,
        waypoint_dim: int,
        hidden_dim: int,
        conv1_channels: int,
        conv2_channels: int,
        max_waypoints: int,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=waypoint_dim,
                out_channels=conv1_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
        )
        self.out = nn.Linear(conv2_channels, hidden_dim // 2)
        # Learnable positional encoding so convolutions can distinguish waypoint position
        self.pos_encoding = nn.Parameter(torch.randn(1, max_waypoints, waypoint_dim) * 0.02)

    def forward(self, flight_plans: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # flight_plans: [batch_size, waypoints, features]
        if mask is None:  # Can happen during dummy shape probing in setup.
            # Derive mask from raw input BEFORE pos_encoding (which makes zero rows nonzero)
            mask = flight_plans.abs().sum(dim=-1) > 0  # [batch_size, waypoints]
        else:
            mask = mask > 0

        # Prevent padded rows from leaking through convolution:
        # mask raw inputs and apply positional encoding only on valid waypoints.
        waypoint_mask = mask.unsqueeze(-1).to(flight_plans.dtype)
        pos = self.pos_encoding[:, : flight_plans.shape[1], :]
        flight_plans = (flight_plans * waypoint_mask) + (pos * waypoint_mask)
        x = flight_plans.transpose(1, 2)  # -> [batch_size, features, waypoints]
        x = self.conv(x).transpose(1, 2)  # -> [batch_size, waypoints, channels]

        # Masked global average pooling over waypoints.
        # Because each waypoint was position-encoded before convolution, the per-position
        # representations already carry order information — averaging them preserves this
        # (same principle as BERT-style mean pooling over position-encoded token outputs).
        x = x.transpose(1, 2)  # -> [batch_size, channels, waypoints]
        mask_expanded = mask.unsqueeze(1).to(x.dtype)  # [batch_size, 1, waypoints]
        x = x * mask_expanded
        denom = mask_expanded.sum(dim=-1).clamp_min(1.0)  # [batch_size, 1]
        x = x.sum(dim=-1) / denom  # [batch_size, channels]
        return F.silu(self.out(x))  # [batch_size, hidden_dim // 2]


class CrossAttentionNeighborEncoder(nn.Module):
    """Ego-query cross-attention encoder for neighbor features."""

    def __init__(
        self,
        ego_dim: int,
        neighbor_feature_dim: int,
        model_dim: int = 64,
        num_heads: int = 1,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                "CrossAttentionNeighborEncoder requires model_dim divisible by num_heads; "
                f"got model_dim={model_dim}, num_heads={num_heads}"
            )
        self.ego_proj = nn.Linear(ego_dim, model_dim)
        self.attn_pre_norm = nn.LayerNorm(model_dim)
        self.attn_post_norm = nn.LayerNorm(model_dim)
        self.null_neighbor = nn.Parameter(torch.zeros(1, 1, neighbor_feature_dim))
        self.neighbor_proj = nn.Sequential(
            nn.Linear(neighbor_feature_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        ego_embedding: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_features, neighbor_mask = append_null_neighbor_feature(
            neighbor_features=neighbor_features,
            neighbor_mask=neighbor_mask,
            null_neighbor=self.null_neighbor,
        )

        encoded = self.neighbor_proj(neighbor_features)  # [B, K+1, D]
        ego = self.ego_proj(ego_embedding)  # [B, D]
        key_padding_mask = neighbor_mask <= 0  # [B, K+1]
        context, _ = self.cross_attn(
            query=self.attn_pre_norm(ego).unsqueeze(1),  # [B, 1, D]
            key=encoded,
            value=encoded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        context = self.attn_post_norm(ego + context.squeeze(1))

        output: torch.Tensor = self.output_proj(torch.cat([ego, context], dim=-1))
        return output
