"""Tests for MAPPO-style value normalization in custom PPO learner."""

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.evaluation.postprocessing import Postprocessing

from config.config import Config
from network.value_normalizer import RunningMeanStd
from training.learner_checkpoint import ATCPPOLearner


def test_running_mean_std_converges_toward_new_distribution() -> None:
    rms = RunningMeanStd()
    for _ in range(50):
        rms.update(torch.tensor([100.0, 100.0, 100.0]))
    mean_before = rms.mean.item()
    for _ in range(250):
        rms.update(torch.tensor([0.0, 0.0, 0.0]))
    assert rms.mean.item() < mean_before * 0.1


def test_running_mean_std_normalize_centers_and_scales() -> None:
    rms = RunningMeanStd()
    rms.update(torch.tensor([10.0, 20.0, 30.0]))
    normalized = rms.normalize(torch.tensor([20.0]))
    assert abs(normalized.item()) < 1.5  # should be near 0 (close to mean)


def test_running_mean_std_state_dict_round_trip() -> None:
    rms = RunningMeanStd()
    rms.update(torch.tensor([10.0, 20.0, 30.0]))
    rms2 = RunningMeanStd()
    rms2.load_state_dict(rms.state_dict())
    assert rms2.mean.item() == rms.mean.item()
    assert rms2.std.item() == rms.std.item()


class _FakeNormalizer:
    def __init__(self) -> None:
        self.update_calls = 0
        self.updated_values: list[torch.Tensor] = []

    @property
    def mean_acc(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def update(self, values: torch.Tensor) -> None:
        self.update_calls += 1
        self.updated_values.append(values.detach().cpu().clone())

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return values - 10.0


class _FakeModule:
    def __init__(self) -> None:
        self.value_normalizer = _FakeNormalizer()
        self._skip_value_denorm = False

    def is_stateful(self):
        return False

    def unwrapped(self):
        return self


class _FakeMultiAgentBatch:
    def __init__(self, policy_batches: dict) -> None:
        self.policy_batches = policy_batches


def test_value_normalizer_updates_once_from_full_train_batch(monkeypatch) -> None:
    learner = ATCPPOLearner.__new__(ATCPPOLearner)
    module = _FakeModule()
    learner._module = {Config.POLICY_NAME: module}

    batch = _FakeMultiAgentBatch(
        {Config.POLICY_NAME: {Postprocessing.VALUE_TARGETS: torch.tensor([1.0, 2.0, 3.0, 4.0])}}
    )

    def fake_make_batch(self, training_data):  # pylint: disable=unused-argument
        return batch

    monkeypatch.setattr(PPOTorchLearner, "_make_batch_if_necessary", fake_make_batch)

    converted = learner._make_batch_if_necessary(training_data=object())

    assert converted is batch
    assert module.value_normalizer.update_calls == 1
    assert torch.equal(
        module.value_normalizer.updated_values[0], torch.tensor([1.0, 2.0, 3.0, 4.0])
    )


def test_compute_loss_normalizes_with_frozen_value_stats(monkeypatch) -> None:
    learner = ATCPPOLearner.__new__(ATCPPOLearner)
    module = _FakeModule()
    learner._module = {Config.POLICY_NAME: module}

    def forbidden_update(_values: torch.Tensor) -> None:
        raise AssertionError("normalizer must not update inside PPO loss calls")

    monkeypatch.setattr(module.value_normalizer, "update", forbidden_update)

    def fake_super_loss(self, *, module_id, config, batch, fwd_out):  # pylint: disable=unused-argument
        assert module._skip_value_denorm
        assert torch.equal(batch[Postprocessing.VALUE_TARGETS], torch.tensor([10.0, 20.0]))
        return torch.tensor(3.0)

    monkeypatch.setattr(PPOTorchLearner, "compute_loss_for_module", fake_super_loss)

    batch = {Postprocessing.VALUE_TARGETS: torch.tensor([20.0, 30.0])}
    loss = learner.compute_loss_for_module(
        module_id=Config.POLICY_NAME,
        config=None,
        batch=batch,
        fwd_out={},
    )

    assert loss.item() == 3.0
    assert not module._skip_value_denorm


def _statistics_learner(*module_ids, stateful=False):
    from types import SimpleNamespace

    learner = ATCPPOLearner.__new__(ATCPPOLearner)
    learner._module = {}
    for module_id in module_ids:
        module = SimpleNamespace(value_normalizer=RunningMeanStd(), is_stateful=lambda: stateful)
        module.unwrapped = lambda module=module: module
        learner._module[module_id] = module
    return learner


def _assert_statistics_equal(actual, expected):
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value)


def test_masked_padding_does_not_change_value_statistics():
    from ray.rllib.core.columns import Columns

    expected = RunningMeanStd()
    expected.update(torch.tensor([10.0]))
    for padding in (0.0, -12345.0, float("nan")):
        learner = _statistics_learner("p", stateful=True)
        learner._update_value_normalizers_from_batches(
            [
                {
                    Postprocessing.VALUE_TARGETS: torch.tensor([[10.0] + [padding] * 7]),
                    Columns.LOSS_MASK: torch.tensor([[1] + [0] * 7]),
                }
            ]
        )
        _assert_statistics_equal(learner.module["p"].value_normalizer, expected)


def test_empty_mask_preserves_existing_statistics():
    from ray.rllib.core.columns import Columns

    learner = _statistics_learner("p", stateful=True)
    normalizer = learner.module["p"].value_normalizer
    normalizer.update(torch.tensor([3.0, 7.0]))
    before = {key: value.clone() for key, value in normalizer.state_dict().items()}
    learner._update_value_normalizers_from_batches(
        [
            {
                Postprocessing.VALUE_TARGETS: torch.full((2, 3), float("nan")),
                Columns.LOSS_MASK: torch.zeros((2, 3), dtype=torch.bool),
            }
        ]
    )
    for key, value in before.items():
        torch.testing.assert_close(normalizer.state_dict()[key], value)


def test_recurrent_row_selection_and_modules_remain_independent():
    from ray.rllib.core.columns import Columns

    learner = _statistics_learner("p", "q", stateful=True)
    learner._update_value_normalizers_from_batches(
        [
            _FakeMultiAgentBatch(
                {
                    "p": {
                        Postprocessing.VALUE_TARGETS: torch.tensor(
                            [[1.0, 900.0, 3.0], [900.0, 5.0, 900.0]]
                        ),
                        Columns.LOSS_MASK: torch.tensor(
                            [[True, False, True], [False, True, False]]
                        ),
                    },
                    "q": {
                        Postprocessing.VALUE_TARGETS: torch.tensor([[900.0, -20.0, -40.0]]),
                        Columns.LOSS_MASK: torch.tensor([[False, True, True]]),
                    },
                }
            ),
            _FakeMultiAgentBatch(
                {
                    "p": {
                        Postprocessing.VALUE_TARGETS: torch.tensor([[7.0, 900.0]]),
                        Columns.LOSS_MASK: torch.tensor([[True, False]]),
                    }
                }
            ),
        ]
    )
    for module_id, targets in (("p", [1.0, 3.0, 5.0, 7.0]), ("q", [-20.0, -40.0])):
        expected = RunningMeanStd()
        expected.update(torch.tensor(targets))
        _assert_statistics_equal(learner.module[module_id].value_normalizer, expected)


def test_incompatible_mask_shape_is_rejected():
    import pytest
    from ray.rllib.core.columns import Columns

    learner = _statistics_learner("p", stateful=True)
    for mask in (torch.ones(6), torch.ones((3, 2)), torch.ones((2, 1))):
        with pytest.raises(ValueError, match="mask.*shape"):
            learner._update_value_normalizers_from_batches(
                [{Postprocessing.VALUE_TARGETS: torch.ones((2, 3)), Columns.LOSS_MASK: mask}]
            )


def test_missing_mask_is_allowed_only_for_nonpadded_targets():
    import pytest

    learner = _statistics_learner("p")
    learner._update_value_normalizers_from_batches(
        [{Postprocessing.VALUE_TARGETS: torch.tensor([2.0, 6.0])}]
    )
    expected = RunningMeanStd()
    expected.update(torch.tensor([2.0, 6.0]))
    _assert_statistics_equal(learner.module["p"].value_normalizer, expected)
    for stateful, targets in ((True, torch.ones(3)), (False, torch.ones((1, 3)))):
        learner = _statistics_learner("p", stateful=stateful)
        with pytest.raises(ValueError, match="loss mask"):
            learner._update_value_normalizers_from_batches(
                [{Postprocessing.VALUE_TARGETS: targets}]
            )
