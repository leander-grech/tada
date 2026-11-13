from math import sqrt

import torch
from config.config import Config
from torch import nn
from torch.nn import functional as F
from torch.nn import init


# ============================================================
# Flight plan encoder
# ============================================================

class FlightPlanConvEncoder(nn.Module):
    """1D CNN encoder for waypoint sequences."""

    def __init__(
        self,
        waypoint_dim: int,
        hidden_dim: int,
        conv1_channels: int,
        conv2_channels: int,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=waypoint_dim,
                out_channels=conv1_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.out = nn.Linear(conv2_channels, hidden_dim // 2)

    def forward(
        self,
        flight_plans: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            flight_plans: [B, W, F]
            mask: [B, W]

        Returns:
            Encoded flight plan: [B, hidden_dim // 2]
        """

        # [B, W, F] -> [B, F, W]
        x = flight_plans.transpose(1, 2)

        x = self.conv(x)

        mask = mask.unsqueeze(1).float()

        x = x * mask

        denom = mask.sum(dim=-1).clamp_min(1.0)

        x = x.sum(dim=-1) / denom

        return F.relu(self.out(x))


# ============================================================
# Shared observation encoder
# ============================================================

class AircraftSelfAttention(nn.Module):
    """Masked self-attention across aircraft slots. Per-slot projections and attention preserve
    permutation equivariance. Invalid slots are excluded as keys. The zero-initialized
    output projection gives LayerNorm(x) initially, rather than an identity mapping;
    preserve this normalization when loading selected checkpoints."""

    def __init__(self, dim: int, attn_dim: int, heads: int):
        super().__init__()
        if attn_dim % heads != 0:
            raise ValueError(f"ATTENTION_DIM ({attn_dim}) must divide ATTENTION_HEADS ({heads})")
        self.heads = heads
        self.head_dim = attn_dim // heads
        self.q = nn.Linear(dim, attn_dim, bias=False)
        self.k = nn.Linear(dim, attn_dim, bias=False)
        self.v = nn.Linear(dim, attn_dim, bias=False)
        self.out = nn.Linear(attn_dim, dim)
        self.norm = nn.LayerNorm(dim)
        for lin in (self.q, self.k, self.v):
            nn.init.orthogonal_(lin.weight, gain=1.0)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, mask_aircraft: torch.Tensor) -> torch.Tensor:
        b, a, _ = x.shape
        h, d = self.heads, self.head_dim
        shape = lambda t: t.view(b, a, h, d).transpose(1, 2)      # [B, H, A, d]
        q, k, v = shape(self.q(x)), shape(self.k(x)), shape(self.v(x))

        scores = (q @ k.transpose(-2, -1)) / sqrt(d)          # [B, H, A, A]
        keep = mask_aircraft.bool()[:, None, None, :]              # [B, 1, 1, A]
        scores = scores.masked_fill(~keep, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # A row whose keys are all masked softmaxes to NaN. That needs a live aircraft to
        # occur at all, but zero it rather than let a NaN reach the heads.
        attn = torch.nan_to_num(attn, nan=0.0)

        pooled = (attn @ v).transpose(1, 2).reshape(b, a, h * d)   # [B, A, attn_dim]
        return self.norm(x + self.out(pooled))


class ATCEncoder(nn.Module):
    """Shared obs encoder used by both the flat feature extractor and the
    autoregressive policy network.

    `encode(obs)` returns:
    - aircraft_emb:  [B, A, H]   fused per-aircraft embeddings (pre-pool)
    - mask_aircraft: [B, A]      float aircraft-validity mask
    - context:       [B, 2H]     concat(masked-mean pooled aircraft, global embedding)

    The flat path pools `aircraft_emb` away (uses only `context`); the autoregressive
    path conditions its clearance head on the per-aircraft embeddings.
    """

    def __init__(self):
        super().__init__()

        hidden_units = int(Config.AR_HIDDEN)
        aircraft_feature_dim = int(Config.AIRCRAFT_SCALAR_LEN)
        global_dim = int(Config.GLOBAL_DIM)
        waypoint_dim = int(Config.FLIGHT_PLAN_WAYPOINT_DIM)
        action_history_dim = 1 + int(Config.N_CLEARANCES)

        self.hidden_units = hidden_units

        relu_gain = float(sqrt(2.0))

        # Aircraft scalar encoder
        self.ac_mlp = nn.Sequential(
            nn.Linear(aircraft_feature_dim, hidden_units),
            nn.LayerNorm(hidden_units),
            nn.ReLU(),
        )

        # Global encoder
        self.glob_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_units),
            nn.ReLU(),
        )

        # Flight plan encoder
        self.fp_encoder = FlightPlanConvEncoder(
            waypoint_dim=waypoint_dim,
            hidden_dim=hidden_units,
            conv1_channels=int(Config.CONV1_CHANNELS),
            conv2_channels=int(Config.CONV2_CHANNELS),
        )

        # Per-aircraft action history encoder
        self.action_history_gru = nn.GRU(
            input_size=action_history_dim,
            hidden_size=hidden_units // 2,
            batch_first=True,
        )

        # Per-aircraft fusion:
        #   aircraft scalar (hidden) + flight plan (hidden//2) + action history (hidden//2)
        self.aircraft_fuse = nn.Linear(
            hidden_units + (hidden_units // 2) + (hidden_units // 2),
            hidden_units,
        )
        self.aircraft_fuse_norm = nn.LayerNorm(hidden_units)

        self.aircraft_attention = (
            AircraftSelfAttention(hidden_units,
                                  int(Config.ATTENTION_DIM),
                                  int(Config.ATTENTION_HEADS))
            if bool(getattr(Config, "USE_AIRCRAFT_ATTENTION", False)) else None
        )

        # Init
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=relu_gain)
                if module.bias is not None:
                    init.zeros_(module.bias)

        # The sweep above touches every nn.Linear, including the attention block's, which undoes
        # its zero-initialised output projection. Restore it: the residual branch must start at
        # exactly zero so the block is an identity no-op on step 0 and the run begins in the same
        # function class as every previous one.
        if self.aircraft_attention is not None:
            init.zeros_(self.aircraft_attention.out.weight)
            init.zeros_(self.aircraft_attention.out.bias)

        for name, param in self.action_history_gru.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)
            elif "bias" in name:
                init.zeros_(param)

    @property
    def aircraft_dim(self) -> int:
        """Per-aircraft embedding dimension."""
        return self.hidden_units

    @property
    def context_dim(self) -> int:
        """Pooled-aircraft + global context dimension."""
        return self.hidden_units * 2

    def encode(self, obs):
        aircraft = obs["aircraft"].float()
        global_features = obs["global"].float()
        flight_plans_rel = obs["flight_plans_rel"].float()
        flight_plan_mask = obs["flight_plan_mask"]
        mask_aircraft = obs["mask_aircraft"]
        action_histories = obs["action_histories"].float()

        batch_size, num_aircraft, num_waypoints, waypoint_dim = flight_plans_rel.shape

        # Aircraft scalar encoding -> [B, A, hidden]
        aircraft_embeddings = self.ac_mlp(aircraft)

        # Flight plan encoding -> [B, A, hidden // 2]
        flight_plans_2d = flight_plans_rel.reshape(
            batch_size * num_aircraft, num_waypoints, waypoint_dim
        )
        flight_plan_mask_2d = flight_plan_mask.reshape(
            batch_size * num_aircraft, num_waypoints
        )
        fp_emb_2d = self.fp_encoder(flight_plans_2d, flight_plan_mask_2d)
        fp_emb = fp_emb_2d.reshape(batch_size, num_aircraft, -1)

        # Action history encoding -> [B, A, hidden // 2]
        _, _, history_len, action_history_dim = action_histories.shape
        action_histories_2d = action_histories.reshape(
            batch_size * num_aircraft, history_len, action_history_dim
        )
        _, h_n = self.action_history_gru(action_histories_2d)
        action_hist_emb = h_n[-1].reshape(batch_size, num_aircraft, -1)

        # Fuse aircraft scalar + flight plan + action history -> [B, A, hidden]
        fused_aircraft = torch.cat(
            [aircraft_embeddings, fp_emb, action_hist_emb], dim=-1
        )
        aircraft_embeddings = self.aircraft_fuse(fused_aircraft)
        aircraft_embeddings = self.aircraft_fuse_norm(aircraft_embeddings)
        aircraft_embeddings = F.relu(aircraft_embeddings)

        mask_aircraft = mask_aircraft.float()

        # Let each aircraft read the others before anything is pooled, so BOTH the per-aircraft
        # embeddings the heads score and the pooled context carry pairwise structure.
        if self.aircraft_attention is not None:
            aircraft_embeddings = self.aircraft_attention(aircraft_embeddings, mask_aircraft)

        # Aircraft masking + masked-mean pooling
        mask_expanded = mask_aircraft.unsqueeze(-1)
        masked_aircraft = aircraft_embeddings * mask_expanded
        denom = mask_expanded.sum(dim=1).clamp_min(1.0)
        pooled_aircraft = masked_aircraft.sum(dim=1) / denom

        # Global encoding -> [B, hidden]
        global_embedding = self.glob_mlp(global_features)

        # Context -> [B, 2 * hidden]
        context = torch.cat([pooled_aircraft, global_embedding], dim=-1)

        # Return per-aircraft embeddings *before* masking, so the autoregressive
        # heads see every aircraft; masking is applied at the categorical logits.
        return aircraft_embeddings, mask_aircraft, context





# ============================================================
# Autoregressive policy network (MultiDiscrete path)
# ============================================================

class ATCAutoregressiveNetwork(nn.Module):
    """Autoregressive actor-critic network: aircraft head -> clearance head.

    Returns raw (unmasked) logits + value; the policy applies masks (read from the
    observation) at the categorical level so that the clearance distribution can be
    conditioned on the *sampled* aircraft.

    forward(obs) -> (aircraft_logits [B, A], clearance_logits [B, A, C], values [B])

    `clearance_logits` are produced for *every* aircraft so the policy can both gather
    the chosen aircraft's row and compute the exact expected entropy over aircraft.
    """

    def __init__(self, head_hidden: int = 64, again: bool = False):
        super().__init__()

        self.encoder = ATCEncoder()
        self.n_clearances = int(Config.N_CLEARANCES)

        per_ac_in = self.encoder.aircraft_dim + self.encoder.context_dim
        ctx_in = self.encoder.context_dim

        self.aircraft_head = nn.Sequential(
            nn.Linear(per_ac_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.clearance_head = nn.Sequential(
            nn.Linear(per_ac_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.n_clearances),
        )

        self.value_head = nn.Sequential(
            nn.Linear(ctx_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Init: hidden layers with relu gain; small gain on the policy output layers
        # (start near-uniform), unit gain on the value output.
        relu_gain = float(sqrt(2.0))
        for head in (self.aircraft_head, self.clearance_head, self.value_head):
            linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
            for i, m in enumerate(linears):
                is_last = i == len(linears) - 1
                if is_last and head is self.value_head:
                    gain = 1.0
                elif is_last:
                    gain = 0.01
                else:
                    gain = relu_gain
                init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    init.zeros_(m.bias)

        # Reselection (Config.MAX_PICKS > 1): the logit of again=1 for every (aircraft,
        # clearance), so the flag is conditioned on both earlier choices. Built after the other
        # heads so their initialisation is unchanged. Output weights start at zero and the bias
        # at AGAIN_INIT_LOGIT: every flag starts at the same small probability.
        self.again_head = None
        if again:
            self.again_head = nn.Sequential(
                nn.Linear(per_ac_in, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, self.n_clearances),
            )
            init.orthogonal_(self.again_head[0].weight, gain=relu_gain)
            init.zeros_(self.again_head[0].bias)
            init.zeros_(self.again_head[2].weight)
            init.constant_(self.again_head[2].bias, float(Config.AGAIN_INIT_LOGIT))

    def forward_all(self, obs):
        """(aircraft_logits [B, A], clearance_logits [B, A, C], again_logits [B, A, C] or None,
        values [B]), all unmasked."""
        aircraft_emb, _mask_aircraft, context = self.encoder.encode(obs)
        batch_size, num_aircraft, _ = aircraft_emb.shape

        ctx_expanded = context.unsqueeze(1).expand(batch_size, num_aircraft, -1)
        per_ac_in = torch.cat([aircraft_emb, ctx_expanded], dim=-1)

        aircraft_logits = self.aircraft_head(per_ac_in).squeeze(-1)   # [B, A]
        clearance_logits = self.clearance_head(per_ac_in)             # [B, A, C]
        values = self.value_head(context).squeeze(-1)                 # [B]
        again_logits = self.again_head(per_ac_in) if self.again_head is not None else None

        return aircraft_logits, clearance_logits, again_logits, values

    def forward(self, obs):
        aircraft_logits, clearance_logits, _again_logits, values = self.forward_all(obs)
        return aircraft_logits, clearance_logits, values


# ============================================================
# Helper functions
# ============================================================



def decode_action(
    action: int,
):
    """
    Convert flattened action -> (aircraft, clearance).
    """

    aircraft_id = (
        action // int(Config.N_CLEARANCES)
    )

    clearance_id = (
        action % int(Config.N_CLEARANCES)
    )

    return aircraft_id, clearance_id
