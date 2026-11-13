"""Recurrent (GRU) memory and time-dimension folding for the ATC policy network."""

from typing import Any, cast

import torch
from ray.rllib.core.columns import Columns
from torch import nn


class RecurrenceMixin:
    """GRU temporal memory over RLlib state and sequence lengths, with time folding."""

    gru: nn.GRU | None
    gru_norm: nn.LayerNorm | None
    architecture: Any

    @staticmethod
    def _flatten_time(
        t: torch.Tensor, batch_size: int, seq_len: int, has_time_dim: bool
    ) -> torch.Tensor:
        """Flatten time on tensors shaped [B, T, ...] into [B*T, ...]."""
        if not has_time_dim or t.dim() < 3:
            return t
        return t.reshape(batch_size * seq_len, *t.shape[2:])

    @staticmethod
    def _unflatten_time(
        t: torch.Tensor, batch_size: int, seq_len: int, has_time_dim: bool
    ) -> torch.Tensor:
        """Unflatten time dimension: [B*T, D] -> [B, T, D]."""
        if not has_time_dim:
            return t
        return t.reshape(batch_size, seq_len, -1)

    @staticmethod
    def _normalize_gru_state(
        *,
        state_tensor: torch.Tensor | None,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize GRU hidden state to [1, B, H], or initialize zeros when absent."""
        if state_tensor is None:
            return torch.zeros(1, batch_size, hidden_size, device=device, dtype=dtype)

        h = torch.as_tensor(state_tensor, device=device, dtype=dtype)
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)  # [H] -> [1, 1, H]
        elif h.dim() == 2:
            h = h.unsqueeze(0)  # [B, H] -> [1, B, H]
        elif h.dim() == 3:
            # Accept [1, B, H] as-is and normalize [B, 1, H] -> [1, B, H].
            if h.shape[0] != 1 and h.shape[1] == 1:
                h = h.transpose(0, 1)
        else:
            raise ValueError(
                f"STATE_IN['gru_h'] must be [H], [B, H], or [1, B, H] (got shape {tuple(h.shape)})"
            )

        if h.dim() != 3 or h.shape[0] != 1 or h.shape[1] != batch_size or h.shape[2] != hidden_size:
            raise ValueError(
                "STATE_IN['gru_h'] must resolve to [1, B, H] "
                f"with B={batch_size}, H={hidden_size} (got {tuple(h.shape)})"
            )
        return h

    @staticmethod
    def _normalize_seq_lens(
        seq_lens: torch.Tensor | None,
        *,
        batch_size: int,
        max_seq_len: int,
    ) -> torch.Tensor | None:
        """Validate and normalize sequence lengths to CPU LongTensor [B]."""
        if seq_lens is None:
            return None
        if not isinstance(seq_lens, torch.Tensor):
            seq_lens = torch.as_tensor(seq_lens)
        seq_lens = seq_lens.to(device="cpu", dtype=torch.long)
        if seq_lens.dim() != 1 or seq_lens.shape[0] != batch_size:
            raise ValueError(
                "Columns.SEQ_LENS must be [B] and match encoded batch size "
                f"(got shape {tuple(seq_lens.shape)} for batch {batch_size})"
            )
        out_of_range = (seq_lens < 1) | (seq_lens > max_seq_len)
        if bool(out_of_range.any()):
            raise ValueError(
                "Columns.SEQ_LENS values must be within [1, max_seq_len] "
                f"(max_seq_len={max_seq_len}, got min={int(seq_lens.min())}, "
                f"max={int(seq_lens.max())})"
            )
        return seq_lens

    def _apply_gru(
        self,
        encoded: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Apply GRU temporal memory using RLlib state + seq_lens packing."""
        gru = cast(nn.GRU, self.gru)
        gru_norm = cast(nn.LayerNorm, self.gru_norm)

        # RLlib may provide [B, D] or [B, T, D]. Normalize to [B, T, D].
        gru_input = encoded.unsqueeze(1) if encoded.dim() == 2 else encoded
        if gru_input.dim() != 3:
            raise ValueError(
                f"Encoded tensor must be [B, D] or [B, T, D], got {tuple(encoded.shape)}"
            )

        batch_size = gru_input.shape[0]
        hidden_size = int(self.architecture["gru_hidden"])
        state_in = cast(dict[str, torch.Tensor], batch.get(Columns.STATE_IN, {}))
        h = self._normalize_gru_state(
            state_tensor=state_in.get("gru_h"),
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=gru_input.device,
            dtype=gru_input.dtype,
        )

        seq_lens = self._normalize_seq_lens(
            batch.get(Columns.SEQ_LENS),
            batch_size=batch_size,
            max_seq_len=gru_input.size(1),
        )
        if seq_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                gru_input,
                lengths=seq_lens,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, h_new = gru(packed, h)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=gru_input.size(1),
            )
        else:
            gru_out, h_new = gru(gru_input, h)

        gru_out = gru_norm(gru_out)
        state_out = {"gru_h": h_new.squeeze(0)}  # [1, B, H] -> [B, H]
        return gru_out, state_out
