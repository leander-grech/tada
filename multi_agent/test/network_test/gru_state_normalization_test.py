"""Tests for ATCRLM GRU state and sequence-length normalization helpers."""

from __future__ import annotations

import pytest
import torch

from network.rlm import ATCRLM

# pylint: disable=protected-access


def test_normalize_gru_state_none_initializes_zeros() -> None:
    state = ATCRLM._normalize_gru_state(
        state_tensor=None,
        batch_size=3,
        hidden_size=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert state.shape == (1, 3, 5)
    assert torch.all(state == 0.0)


def test_normalize_gru_state_accepts_batch_hidden_tensor() -> None:
    state_in = torch.randn(3, 5)

    state = ATCRLM._normalize_gru_state(
        state_tensor=state_in,
        batch_size=3,
        hidden_size=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert state.shape == (1, 3, 5)
    assert torch.allclose(state.squeeze(0), state_in)


def test_normalize_gru_state_rejects_mismatched_dimensions() -> None:
    state_in = torch.randn(2, 5)

    with pytest.raises(ValueError, match="must resolve to \\[1, B, H\\]"):
        ATCRLM._normalize_gru_state(
            state_tensor=state_in,
            batch_size=3,
            hidden_size=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_normalize_seq_lens_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="must be within \\[1, max_seq_len\\]"):
        ATCRLM._normalize_seq_lens(
            torch.tensor([0, 2, 99], dtype=torch.int64),
            batch_size=3,
            max_seq_len=5,
        )


def test_normalize_seq_lens_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Columns.SEQ_LENS must be \\[B\\]"):
        ATCRLM._normalize_seq_lens(
            torch.tensor([[1, 2]], dtype=torch.int64),
            batch_size=2,
            max_seq_len=5,
        )
