"""GAE's physical termination and valid-only statistics contracts."""

import numpy as np
import pytest


def test_chunk_gae_retains_bootstrap_and_current_terminal_state_value():
    from network.gae import compute_chunk_gae

    for dtype in (np.float32, np.float64):
        for lambda_ in (0.5, 0.97):
            for terminal, expected in ((False, 8.3), (True, 2)):
                targets, advantages = compute_chunk_gae(
                    np.array([3, 7], dtype),
                    np.array([2], dtype),
                    np.array([terminal]),
                    gamma=0.9,
                    lambda_=lambda_,
                )
                np.testing.assert_allclose(targets, [expected])
                np.testing.assert_allclose(advantages, [expected - 3])
                assert targets.dtype == dtype
        targets, _ = compute_chunk_gae(
            np.array([3, 5, 7], dtype),
            np.array([1, 2], dtype),
            np.array([False, True]),
            gamma=0.9,
            lambda_=0.5,
        )
        np.testing.assert_allclose(targets, [4.15, 2])


def test_normalization_ignores_invalid_post_gae_advantages():
    from network.gae import normalize_valid_advantages

    mask = np.array([[True, True, False], [True, False, False]])
    clean = np.array([[1.0, 3.0, 0.0], [5.0, 0.0, 0.0]])
    dirty = clean.copy()
    dirty[~mask] = [np.nan, np.inf, -1e30]
    result = normalize_valid_advantages(dirty, mask)
    np.testing.assert_allclose(result, normalize_valid_advantages(clean, mask))
    np.testing.assert_allclose(result[mask], [-np.sqrt(1.5), 0, np.sqrt(1.5)])
    assert np.array_equal(result[~mask], np.zeros(3))
    assert np.array_equal(
        normalize_valid_advantages(clean, np.zeros_like(mask)), np.zeros_like(clean)
    )


@pytest.mark.parametrize(
    "mutation", ["missing_final", "nonfinite", "bad_flags", "shape", "gamma", "lambda"]
)
def test_chunk_rejects_ambiguous_or_invalid_inputs(mutation):
    from network.gae import compute_chunk_gae

    values, rewards, flags = np.array([3.0, 7.0]), np.array([2.0]), np.array([False])
    gamma, lambda_ = 0.9, 0.5
    if mutation == "missing_final":
        values = values[:-1]
    elif mutation == "nonfinite":
        values[-1] = np.nan
    elif mutation == "bad_flags":
        flags = np.array([1])
    elif mutation == "shape":
        rewards = rewards.reshape(1, 1)
    elif mutation == "gamma":
        gamma = 1.1
    else:
        lambda_ = -0.1
    with pytest.raises(ValueError):
        compute_chunk_gae(values, rewards, flags, gamma=gamma, lambda_=lambda_)
