"""Tests for neighbor encoder behavior."""

import pytest
import torch

from network.encoders import CrossAttentionNeighborEncoder


def test_neighbor_pooling_requires_model_dim_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="model_dim divisible by num_heads"):
        CrossAttentionNeighborEncoder(ego_dim=4, neighbor_feature_dim=3, model_dim=10, num_heads=3)


def test_neighbor_pooling_uses_masked_neighbors_and_outputs_finite_values() -> None:
    torch.manual_seed(0)
    encoder = CrossAttentionNeighborEncoder(ego_dim=4, neighbor_feature_dim=3, model_dim=8)

    ego_embedding = torch.randn(2, 4)
    neighbor_features = torch.randn(2, 4, 3)
    neighbor_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    alt_neighbor_features = neighbor_features.clone()
    alt_neighbor_features[neighbor_mask == 0] = 999.0

    out = encoder(ego_embedding, neighbor_features, neighbor_mask)
    alt_out = encoder(ego_embedding, alt_neighbor_features, neighbor_mask)

    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, alt_out, atol=1e-5, rtol=1e-5)


def test_neighbor_pooling_uses_ego_query_to_weight_neighbors() -> None:
    torch.manual_seed(7)
    encoder = CrossAttentionNeighborEncoder(
        ego_dim=2, neighbor_feature_dim=2, model_dim=8, num_heads=1
    )

    neighbor_features = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    ego_embedding = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    neighbor_mask = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    out = encoder(ego_embedding, neighbor_features, neighbor_mask)

    assert not torch.allclose(out[0], out[1], atol=1e-5, rtol=1e-5)


def test_neighbor_pooling_handles_all_masked_rows_without_nan() -> None:
    torch.manual_seed(0)
    encoder = CrossAttentionNeighborEncoder(ego_dim=4, neighbor_feature_dim=3, model_dim=8)

    ego_embedding = torch.randn(2, 4)
    neighbor_features = torch.randn(2, 4, 3)
    neighbor_mask = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    alt_neighbor_features = neighbor_features.clone()
    alt_neighbor_features[0] = -123.0

    out = encoder(ego_embedding, neighbor_features, neighbor_mask)
    alt_out = encoder(ego_embedding, alt_neighbor_features, neighbor_mask)

    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], alt_out[0], atol=1e-5, rtol=1e-5)


def test_cross_attn_all_masked_rows_backpropagate_null_neighbor_token() -> None:
    torch.manual_seed(0)
    encoder = CrossAttentionNeighborEncoder(
        ego_dim=4,
        neighbor_feature_dim=3,
        model_dim=8,
        num_heads=1,
    )

    ego_embedding = torch.randn(2, 4, requires_grad=True)
    neighbor_features = torch.randn(2, 5, 3)
    neighbor_mask = torch.zeros(2, 5)

    loss = encoder(ego_embedding, neighbor_features, neighbor_mask).sum()
    loss.backward()

    grad = getattr(encoder, "null_neighbor").grad
    assert grad is not None
    assert torch.any(grad != 0)
    assert _has_nonzero_grad_on_non_null_parameters(encoder)


def _set_null_neighbor_value(encoder: torch.nn.Module, value: float) -> None:
    assert hasattr(encoder, "null_neighbor")
    with torch.no_grad():
        encoder.null_neighbor.fill_(value)  # ty: ignore[call-non-callable]


def _has_nonzero_grad_on_non_null_parameters(module: torch.nn.Module) -> bool:
    for name, parameter in module.named_parameters():
        if name == "null_neighbor":
            continue
        grad = parameter.grad
        if grad is not None and torch.any(grad != 0):
            return True
    return False


def test_null_neighbor_is_ignored_when_real_neighbors_exist() -> None:
    torch.manual_seed(0)
    encoder = CrossAttentionNeighborEncoder(
        ego_dim=4,
        neighbor_feature_dim=3,
        model_dim=8,
        num_heads=1,
    )

    ego_embedding = torch.randn(2, 4)
    neighbor_features = torch.randn(2, 5, 3)
    neighbor_mask = torch.ones(2, 5)

    _set_null_neighbor_value(encoder, 111.0)
    out = encoder(ego_embedding, neighbor_features, neighbor_mask)
    _set_null_neighbor_value(encoder, -222.0)
    alt_out = encoder(ego_embedding, neighbor_features, neighbor_mask)

    assert torch.allclose(out, alt_out, atol=1e-5, rtol=1e-5)


def test_null_neighbor_is_used_for_all_masked_rows() -> None:
    torch.manual_seed(0)
    encoder = CrossAttentionNeighborEncoder(
        ego_dim=4,
        neighbor_feature_dim=3,
        model_dim=8,
        num_heads=1,
    )

    ego_embedding = torch.randn(2, 4)
    neighbor_features = torch.randn(2, 5, 3)
    neighbor_mask = torch.zeros(2, 5)

    _set_null_neighbor_value(encoder, 0.0)
    out = encoder(ego_embedding, neighbor_features, neighbor_mask)
    _set_null_neighbor_value(encoder, 5.0)
    alt_out = encoder(ego_embedding, neighbor_features, neighbor_mask)

    assert not torch.allclose(out, alt_out, atol=1e-5, rtol=1e-5)
