"""Config-built owned PPO learner: module values, real chunks and padding."""

import gymnasium as gym
import numpy as np
import pytest
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from support.bootstrap import episode, learner_config, run_pipeline


@pytest.mark.parametrize(
    "mode,expected", [("truncated", 8.3), ("unfinished", 8.3), ("terminal", 2.0)]
)
@pytest.mark.parametrize("lambda_", [0.5, 0.97])
def test_configured_learner_one_step_bootstraps_actual_final_observation(mode, expected, lambda_):
    learner = learner_config(lambda_=lambda_).build_learner()
    result = run_pipeline(learner, [episode(mode=mode)])["p"]
    np.testing.assert_allclose(result[Postprocessing.VALUE_TARGETS].numpy(), [expected, 0])
    np.testing.assert_array_equal(result[Columns.LOSS_MASK].numpy(), [True, False])
    np.testing.assert_array_equal(result[Postprocessing.ADVANTAGES].numpy(), [0, 0])
    assert learner.module["p"].value_calls == 1
    np.testing.assert_array_equal(learner.module["p"].observed_values[0], [3, 7])


@pytest.mark.parametrize(
    "mode,lambda_,expected",
    [
        ("terminal", 0.5, [6.3775, 6.95, 4]),
        ("terminal", 0.97, [6.094513, 5.681, 4]),
        ("truncated", 0.5, [8.38225, 11.405, 13.9]),
        ("truncated", 0.97, [13.6395901, 14.3237, 13.9]),
        ("unfinished", 0.5, [8.38225, 11.405, 13.9]),
        ("unfinished", 0.97, [13.6395901, 14.3237, 13.9]),
    ],
)
@pytest.mark.parametrize("lookback", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_actual_sequence_split_and_cut_exclude_lookback_and_artificial_transitions(
    mode, lambda_, expected, lookback, shared
):
    learner = learner_config(stateful=True, shared=shared, lambda_=lambda_).build_learner()
    eps = [
        episode(
            (3, 5, 7, 11), (1, 2, 4), mode=mode, agents=("a", "b"), shared=shared, lookback=lookback
        ),
        episode((11, 17), (5,), mode="terminal", agents=("a",), shared=shared),
    ]
    batch = run_pipeline(learner, eps)
    for mid, data in batch.items():
        mask = data[Columns.LOSS_MASK].numpy()
        targets = data[Postprocessing.VALUE_TARGETS].numpy()
        expected_rows = [expected, [0, 0, 0]] * (2 if shared else 1) + (
            [[5, 0, 0]] if mid == "p" else []
        )
        np.testing.assert_allclose(targets, expected_rows, rtol=2e-6)
        assert np.all(targets[~mask] == 0)
        advantages = data[Postprocessing.ADVANTAGES].numpy()
        assert np.all(advantages[~mask] == 0)
        # Independently derive raw advantages from the observation values.
        values = data[Columns.OBS].numpy()[..., 0]
        raw = targets[mask] - values[mask]
        np.testing.assert_allclose(
            advantages[mask], (raw - raw.mean()) / max(1e-4, raw.std()), rtol=2e-6, atol=2e-6
        )
        assert learner.module[mid].value_calls == 1


def test_two_step_physical_lambda_return_keeps_preterminal_value():
    learner = learner_config().build_learner()
    batch = run_pipeline(learner, [episode((3, 5, 7), (1, 2), mode="terminal")])["p"]
    np.testing.assert_allclose(batch[Postprocessing.VALUE_TARGETS].numpy(), [4.15, 2, 0])


@pytest.mark.parametrize("mode", ["terminal", "truncated", "unfinished"])
def test_final_observation_is_bootstrap_input_but_padding_is_not(mode, monkeypatch):
    import torch

    learner = learner_config(stateful=True).build_learner()
    module = learner.module["p"]
    original = module.compute_values
    shift = 0.0

    def compute_values(batch, embeddings=None):
        result = original(batch, embeddings=embeddings).clone()
        # Three real rows, final observation alone in the next sequence.
        result[1, 0] += shift
        result[1, 1:] = torch.tensor([float("nan"), float("inf")])
        return result

    monkeypatch.setattr(module, "compute_values", compute_values)
    first = run_pipeline(learner, [episode((3, 5, 7, 11), (1, 2, 4), mode=mode)])["p"]
    shift = 10.0
    second = run_pipeline(learner, [episode((3, 5, 7, 11), (1, 2, 4), mode=mode)])["p"]
    target1, target2 = (b[Postprocessing.VALUE_TARGETS].numpy() for b in (first, second))
    if mode == "terminal":
        np.testing.assert_array_equal(target1, target2)
    else:
        np.testing.assert_allclose(target2[0] - target1[0], [1.8225, 4.05, 9], rtol=2e-6)
    assert np.isfinite(target2).all()
    assert np.all(target2[1] == 0)


@pytest.mark.parametrize(
    "corruption", ["mask", "value_shape", "final_value", "column_shape", "termination", "mapping"]
)
def test_connector_rejects_corrupted_row_layout_before_loss(corruption, monkeypatch):
    import torch

    learner = learner_config().build_learner()
    pipeline = learner._learner_connector
    connector = pipeline.connectors[-1]
    pipeline.remove(type(connector))
    eps = [episode()]
    shared_data = {}
    batch = pipeline(rl_module=learner.module, batch={}, episodes=eps, shared_data=shared_data)
    data = batch["p"]
    if corruption == "mapping":
        shared_data.pop("memorized_map_structure")
    elif corruption == "mask":
        data[Columns.LOSS_MASK][1] = True
    elif corruption == "column_shape":
        data[Columns.REWARDS] = data[Columns.REWARDS][:-1]
    elif corruption == "termination":
        data[Columns.TERMINATEDS][0] = True
    else:
        monkeypatch.setattr(
            learner.module["p"],
            "compute_values",
            lambda *args, **kwargs: (
                torch.tensor([3.0])
                if corruption == "value_shape"
                else torch.tensor([3.0, float("nan")])
            ),
        )
    with pytest.raises(ValueError):
        connector(rl_module=learner.module, batch=batch, episodes=eps, shared_data=shared_data)


def test_custom_connector_stack_and_unknown_ray_version_fail_closed(monkeypatch):
    import ray
    from ray.rllib.connectors.learner import GeneralAdvantageEstimation

    config = learner_config()
    config.learners(
        learner_connector=lambda *_: [GeneralAdvantageEstimation(gamma=0.9, lambda_=0.5)]
    )
    with pytest.raises(ValueError, match="stack"):
        config.build_learner()
    monkeypatch.setattr(ray, "__version__", "unsupported")
    with pytest.raises(ValueError, match="Ray"):
        learner_config().build_learner()


@pytest.mark.parametrize("shared", [True, False])
def test_unequal_agent_lengths_and_values_follow_recorded_mapping(shared):
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
    from support.bootstrap import ACT_SPACE, OBS_SPACE, mapping, shared_mapping

    ep = MultiAgentEpisode(
        observation_space=gym.spaces.Dict({a: OBS_SPACE for a in ("a", "b")}),
        action_space=gym.spaces.Dict({a: ACT_SPACE for a in ("a", "b")}),
        agent_to_module_mapping_fn=shared_mapping if shared else mapping,
    )
    ep.add_env_reset(
        observations={"a": np.array([3.0], np.float32), "b": np.array([13.0], np.float32)}, infos={}
    )
    ep.add_env_step(
        observations={"a": np.array([5.0], np.float32), "b": np.array([17.0], np.float32)},
        actions={"a": 0, "b": 0},
        rewards={"a": 1.0, "b": 5.0},
        terminateds={"b": True, "__all__": False},
        truncateds={"__all__": False},
        infos={},
    )
    ep.add_env_step(
        observations={"a": np.array([7.0], np.float32)},
        actions={"a": 0},
        rewards={"a": 2.0},
        infos={},
    )
    ep.add_env_step(
        observations={"a": np.array([11.0], np.float32)},
        actions={"a": 0},
        rewards={"a": 4.0},
        terminateds={"__all__": False},
        truncateds={"__all__": True},
        infos={},
    )
    learner = learner_config(stateful=True, shared=shared).build_learner()
    shared_data = {}
    batch = run_pipeline(learner, [ep.to_numpy()], shared_data=shared_data)
    expected_values = {"a": [3, 5, 7], "b": [13]}
    expected_targets = {"a": [8.38225, 11.405, 13.9], "b": [5]}
    for mid, data in batch.items():
        order = shared_data["memorized_map_structure"][mid]
        values = [v for _, aid in order for v in expected_values[aid]]
        targets = [v for _, aid in order for v in expected_targets[aid]]
        mask = data[Columns.LOSS_MASK].numpy()
        np.testing.assert_array_equal(data[Columns.OBS].numpy()[..., 0][mask], values)
        np.testing.assert_allclose(
            data[Postprocessing.VALUE_TARGETS].numpy()[mask], targets, rtol=2e-6
        )
        raw = np.array(targets) - values
        np.testing.assert_allclose(
            data[Postprocessing.ADVANTAGES].numpy()[mask],
            (raw - raw.mean()) / max(1e-4, raw.std()),
            atol=1e-6,
        )
