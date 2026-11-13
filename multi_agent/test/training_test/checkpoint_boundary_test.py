"""Completed scenarios, learned receipts and process RNG define a safe boundary."""

import copy
import json
import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from support.checkpoints import make_runner

from network.rlm import ATCRLM
from training.learner_checkpoint import ATCPPOLearner

pytestmark = pytest.mark.usefixtures("ray_env_registry")


@pytest.mark.parametrize("length", [2, 5])
def test_runner_complete_and_partial_drain_has_no_reset_or_transition_loss(length):
    from training.checkpoint_boundary import CheckpointBoundaryError

    _, runner = make_runner(length)
    try:
        env = runner.env.envs[0].unwrapped
        assert env.resets == 0
        complete = runner.sample()
        with pytest.raises(CheckpointBoundaryError, match="metrics"):
            runner.capture_boundary_state()
        runner.get_metrics()
        state = runner.capture_boundary_state()
        assert state["emitted"]["env_steps"] == length
        assert env.resets == 1
        part = runner.sample(num_timesteps=1)
        runner.get_metrics()
        with pytest.raises(CheckpointBoundaryError, match="episode|cached"):
            runner.capture_boundary_state()
        tail = runner.sample(num_timesteps=length - 1)
        runner.get_metrics()
        final = runner.capture_boundary_state()
        assert final["emitted"]["env_steps"] == 2 * length
        assert sum(len(ep) for ep in complete + part + tail) == 2 * length
        assert env.resets == 2
        assert all(ep.is_done for ep in complete + tail)
        assert part[0].is_done is False
    finally:
        runner.stop()


def test_runner_rejects_unknown_connector_and_pending_cache():
    from training.checkpoint_boundary import CheckpointBoundaryError

    _, runner = make_runner(2)
    try:
        runner.sample()
        runner.get_metrics()
        runner._cached_to_module = {"pending": 1}
        with pytest.raises(CheckpointBoundaryError, match="cached"):
            runner.capture_boundary_state()
        runner._cached_to_module = {}
        runner._env_to_module.connectors.reverse()
        with pytest.raises(CheckpointBoundaryError, match="connector"):
            runner.capture_boundary_state()
    finally:
        runner.stop()


def test_process_rng_roundtrip_preserves_next_draws_and_rejects_unknown_state():
    from training.checkpoint_boundary import capture_process_rng, restore_process_rng

    original = capture_process_rng()
    try:
        state = copy.deepcopy(capture_process_rng())
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_process_rng(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        assert expected[:2] == actual[:2]
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        with pytest.raises(ValueError, match="RNG"):
            restore_process_rng(dict(state, unknown=True))
    finally:
        restore_process_rng(original)


def test_phase_with_new_aircraft_count_restores_stream_and_generates_new_count(native_run):
    from dataclasses import replace

    from training.env import TrainingATCEnv
    from training.phase import PhaseTransition, apply_sets
    from training.ppo_config import build_ppo_config

    count = native_run.scenario["aircraft_count"] + 1
    phased, changes = apply_sets(replace(native_run, phase_id="next"), [f"aircraft_count={count}"])
    transition = PhaseTransition.between(native_run, phased, changes)
    original = TrainingATCEnv(build_ppo_config(native_run).env_config)
    restored = TrainingATCEnv(build_ppo_config(phased).env_config)
    try:
        original._runtime_activated = True
        original.reset()
        original.step({aid: 0 for aid in original.agents})
        saved = original.capture_boundary_state()
        with pytest.raises(ValueError, match="identity"):
            restored.restore_boundary_state(saved)
        restored.restore_boundary_state(saved, transition=transition)
        restored._runtime_activated = True
        original.reset()
        restored.reset()
        assert restored.scenario_seed == original.scenario_seed
        assert restored.simulator.world is not None
        assert len(restored.simulator.world.aircraft) == count
    finally:
        original.close()
        restored.close()


def test_native_environment_boundary_restores_stream_before_generation(native_run, monkeypatch):
    from training.env import TrainingATCEnv
    from training.ppo_config import build_ppo_config

    env_config = build_ppo_config(native_run).env_config
    original, restored = TrainingATCEnv(env_config), TrainingATCEnv(env_config)
    try:
        assert original.capture_boundary_state()["boundary"] == "pristine"
        original._runtime_activated = True
        import training.seed_stream as seed_module

        digest = seed_module.content_digest
        reserved = next(iter(native_run.reserved_seeds))

        def force_collision(value):
            if (
                isinstance(value, list)
                and value
                and value[0] == seed_module.ALGORITHM
                and value[-1] == 0
            ):
                return f"{reserved:016x}" + "0" * 48
            return digest(value)

        with monkeypatch.context() as forced:
            forced.setattr(seed_module, "content_digest", force_collision)
            original.reset()
        assert original._training_stream is not None
        assert original._training_stream.rejected == 1
        assert original.scenario_seed not in native_run.reserved_seeds
        original.step({aid: 0 for aid in original.agents})
        saved = original.capture_boundary_state()
        assert saved["audit"]["last_end_reason"] == original.last_end_reason
        assert saved["audit"]["world_sha256"]
        incompatible = copy.deepcopy(saved)
        incompatible["stream"]["stream"]["domain"] = "operational_evaluation"
        with pytest.raises(ValueError, match="identity"):
            restored.restore_boundary_state(incompatible)
        restored.restore_boundary_state(saved)
        assert restored.simulator.world is None
        assert restored._has_reset_once is False
        with pytest.raises(RuntimeError, match="activation"):
            restored.reset()
        restored._runtime_activated = True
        original.reset()
        restored.reset()
        assert restored.scenario_seed == original.scenario_seed
        assert restored.simulator.world is not None and original.simulator.world is not None
        assert json.loads(restored.simulator.world.json()) == json.loads(
            original.simulator.world.json()
        )
        with pytest.raises(ValueError, match="first reset"):
            restored.restore_boundary_state(saved)
    finally:
        original.close()
        restored.close()


def test_receipts_reject_unlearned_gaps_duplicates_and_transport_slices():
    from training.checkpoint_boundary import advance_learned_receipts

    _, runner = make_runner(3)
    try:
        first = runner.sample(num_timesteps=1)
        rest = runner.sample(num_timesteps=2)
        with pytest.raises(ValueError, match="noncontiguous"):
            advance_learned_receipts({}, rest)
        learned = advance_learned_receipts({}, first + rest)
        assert learned[runner.checkpoint_identity()] == {"sequence": 2, "env_steps": 3}
        with pytest.raises(ValueError, match="duplicate"):
            advance_learned_receipts(learned, rest)
        sliced = copy.deepcopy(rest)
        sliced[0].custom_data["atc_checkpoint_receipt_v1"]["end"] += 1
        with pytest.raises(ValueError, match="transported"):
            advance_learned_receipts({}, sliced)
    finally:
        runner.stop()


def test_native_learner_state_first_loss_and_phase_update(native_run):
    from ray.rllib.env.env_context import EnvContext

    from training.env_runner import ATCEnvRunner
    from training.ppo_config import build_ppo_config

    cfg = build_ppo_config(native_run).learners(num_learners=0)
    cfg.env_config = EnvContext(cfg.env_config, worker_index=1)
    gym.envs.registration.registry.pop("rllib-multi-agent-env-v0", None)
    runner = ATCEnvRunner(cfg, worker_index=1)
    try:
        runner.activate_environments(native_run.semantic_digest)
        assert runner.env is not None and runner.env.envs is not None
        env = runner.env.envs[0].unwrapped
        reset_counts, sample_counts = [], []
        reset = env.reset

        def counted_reset(**kwargs):
            result = reset(**kwargs)
            reset_counts.append((env.scenario_seed, len(result[0])))
            return result

        env.reset = counted_reset
        learner = cfg.build_learner(env=env)
        assert isinstance(learner, ATCPPOLearner)

        def sample_batch():
            episodes = []
            while sum(ep.agent_steps() for ep in episodes) < 8:
                sampled = runner.sample()
                sample_counts.extend((ep.env_steps(), ep.agent_steps()) for ep in sampled)
                episodes.extend(sampled)
            runner.get_metrics()
            return episodes

        episodes = sample_batch()
        learner.update(episodes=episodes, num_epochs=1)
        saved = copy.deepcopy(learner.get_state())
        identity = runner.checkpoint_identity()
        assert saved["atc_effective_state"]["learned"][identity] == runner._checkpoint_emitted
        model = learner.module["shared_policy"]
        restored = cfg.build_learner(env=runner.env.envs[0].unwrapped)
        assert isinstance(restored, ATCPPOLearner)
        restored.set_state(saved)
        restored_model = restored.module["shared_policy"]
        assert isinstance(model, ATCRLM) and isinstance(restored_model, ATCRLM)
        assert restored_model.value_normalizer is not None
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored_model.state_dict()[name], value)
        seen = []
        loss = restored.compute_loss_for_module

        def inspect_loss(**kwargs):
            seen.append(restored._effective_state()["modules"]["shared_policy"])
            return loss(**kwargs)

        restored.compute_loss_for_module = inspect_loss
        restored.update(episodes=sample_batch(), num_epochs=1)
        expected_constants = saved["atc_effective_state"]["modules"]["shared_policy"]
        assert seen and all(item == expected_constants for item in seen)
        seen.clear()
        normalizer_before = copy.deepcopy(restored_model.value_normalizer.state_dict())
        restored.apply_phase_overrides(
            {"learning_rate": 0.000123456789, "entropy_coeff": 0.023456789}
        )
        for name, value in normalizer_before.items():
            torch.testing.assert_close(restored_model.value_normalizer.state_dict()[name], value)
        episodes = sample_batch()
        restored.update(episodes=episodes, num_epochs=2)
        assert seen and all(
            item == {"learning_rate": 0.000123456789, "entropy_coeff": 0.023456789} for item in seen
        )
        state = restored.get_state()["atc_effective_state"]
        assert state["updates"] == 3
        assert state["learned"][identity] == runner._checkpoint_emitted
        assert runner.capture_boundary_state()["environment"]["boundary"] == "closed"
        from common.runtime_log import runtime_print

        runtime_print(
            "Checkpoint", "native_reset_counts", resets=reset_counts, samples=sample_counts
        )
        assert len(reset_counts) == len(sample_counts)
    finally:
        runner.stop()


def test_recurrent_different_length_runners_drain_through_actual_learning():
    cfg, short = make_runner(2, stateful=True, worker_index=1)
    _, long = make_runner(5, stateful=True, worker_index=2)
    try:
        learner = cfg.build_learner()
        first = short.sample(num_timesteps=1) + long.sample(num_timesteps=1)
        learner.update(episodes=copy.deepcopy(first), num_epochs=2)
        for runner in (short, long):
            runner.get_metrics()
            with pytest.raises(ValueError, match="episode"):
                runner.capture_boundary_state()
        tail = short.sample(num_timesteps=1) + long.sample(num_timesteps=4)
        learner.update(episodes=copy.deepcopy(tail), num_epochs=2)
        for runner, length in ((short, 2), (long, 5)):
            runner.get_metrics()
            state = runner.capture_boundary_state()
            assert learner._checkpoint_learned[runner.checkpoint_identity()] == state["emitted"]
            assert state["emitted"]["env_steps"] == length
            assert runner.env.envs[0].unwrapped.resets == 1
        assert learner._checkpoint_updates == 2
    finally:
        short.stop()
        long.stop()


@pytest.mark.parametrize("epochs,minibatch_size,expected_work", [(1, 3, 3), (2, 6, 6)])
def test_recurrent_minibatch_never_optimizes_artificial_only_sequence(
    epochs, minibatch_size, expected_work
):
    from ray.rllib.core.columns import Columns
    from ray.rllib.evaluation.postprocessing import Postprocessing

    from common.runtime_log import runtime_print
    from network.value_normalizer import RunningMeanStd

    cfg, runner = make_runner(3, stateful=True, worker_index=1)
    try:
        learner = cfg.build_learner()
        episodes = runner.sample()
        runner.get_metrics()
        observed = []
        module = learner.module["p"]
        original_prepare = learner._make_batch_if_necessary
        prepared = []

        def record_prepared(training_data):
            batch = original_prepare(training_data)
            data = batch.policy_batches["p"]
            prepared.append(copy.deepcopy(data))
            return batch

        learner._make_batch_if_necessary = record_prepared
        original_loss = learner.compute_loss_for_module

        def record_loss(**kwargs):
            mask = kwargs["batch"][Columns.LOSS_MASK]
            loss = original_loss(**kwargs)
            observed.append((mask.tolist(), float(loss.detach())))
            return loss

        learner.compute_loss_for_module = record_loss
        result = learner.update(
            episodes=episodes,
            num_epochs=epochs,
            minibatch_size=minibatch_size,
            shuffle_batch_per_epoch=False,
        )
        runtime_print("AuditA", "recurrent minibatch losses", observed=observed)
        assert observed and all(np.isfinite(loss) for _, loss in observed), observed
        assert all(any(any(row) for row in mask) for mask, _ in observed), observed
        assert sum(sum(sum(row) for row in mask) for mask, _ in observed) == expected_work
        assert len(prepared) == 1
        batch = prepared[0]
        assert batch[Columns.LOSS_MASK].tolist() == [[True, True, True]]
        assert batch[Columns.SEQ_LENS].tolist() == [3]
        assert batch[Columns.OBS].squeeze(-1).tolist() == [[3.0, 7.0, 7.0]]
        assert batch[Columns.STATE_IN]["h"].shape == (1, 1)
        # The fourth observation must have participated in the legitimate GAE bootstrap.
        assert module.observed_values[0].tolist() == [[3.0, 7.0, 7.0], [7.0, 0.0, 0.0]]
        targets = torch.tensor([[7.49575, 7.435, 7.3]])
        torch.testing.assert_close(batch[Postprocessing.VALUE_TARGETS], targets)
        expected = RunningMeanStd()
        expected.update(targets.flatten())
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(module.value_normalizer.state_dict()[key], value)
        assert all(torch.isfinite(value).all() for value in module.state_dict().values())
        optimizer = next(iter(learner._named_optimizers.values()))
        assert optimizer.state
        for slots in optimizer.state.values():
            assert all(torch.isfinite(value).all() for value in slots.values())
            assert slots["step"].item() == len(observed)
        assert (
            result["__all_modules__"]["actual_valid_optimizer_rows_this_update"].peek()
            == expected_work
        )
        assert result["__all_modules__"]["actual_optimizer_minibatches_this_update"].peek() == len(
            observed
        )
        assert result["__all_modules__"]["actual_optimizer_updates_this_update"].peek() == 1
        assert learner._checkpoint_updates == 1
        assert (
            learner._checkpoint_learned[runner.checkpoint_identity()] == runner._checkpoint_emitted
        )
        second = learner.update(
            episodes=runner.sample(),
            num_epochs=epochs,
            minibatch_size=minibatch_size,
            shuffle_batch_per_epoch=False,
        )
        for name in (
            "actual_valid_optimizer_rows_this_update",
            "actual_optimizer_minibatches_this_update",
            "actual_optimizer_updates_this_update",
        ):
            assert second["__all_modules__"][name].peek() == result["__all_modules__"][name].peek()
        assert learner._checkpoint_updates == 2
        assert learner._checkpoint_learned[runner.checkpoint_identity()]["env_steps"] == 6
        learner.after_gradient_based_update(timesteps={})
        # A previously reduced Ray sum has no samples (peek is NaN), not zero.
        assert np.isnan(
            learner.metrics.peek(("__all_modules__", "actual_optimizer_updates_this_update"))
        )
    finally:
        runner.stop()


def test_recurrent_minibatch_rejection_precedes_learner_mutation():
    cfg, runner = make_runner(3, stateful=True, worker_index=1)
    try:
        learner = cfg.build_learner()
        episodes = runner.sample()
        before = copy.deepcopy(learner.get_state())
        module = learner.module["p"]
        weights_seq_no = learner._weights_seq_no
        with pytest.raises(ValueError, match="minibatch"):
            learner.update(episodes=episodes, num_epochs=2, minibatch_size=1)
        assert learner._weights_seq_no == weights_seq_no
        assert learner.get_state()["atc_effective_state"] == before["atc_effective_state"]
        for key, value in before["rl_module"]["p"].items():
            np.testing.assert_array_equal(learner.get_state()["rl_module"]["p"][key], value)
        assert not next(iter(learner._named_optimizers.values())).state
        assert module.value_calls == 0
        assert (
            learner.metrics.peek(
                ("__all_modules__", "actual_optimizer_updates_this_update"), default=0
            )
            == 0
        )
        assert (
            learner.metrics.peek(
                ("__all_modules__", "actual_optimizer_minibatches_this_update"), default=0
            )
            == 0
        )
        assert len(episodes[0]) == 3
    finally:
        runner.stop()


def test_recurrent_minibatch_failed_optimizer_is_not_acknowledged(monkeypatch):
    cfg, runner = make_runner(3, stateful=True, worker_index=1)
    try:
        learner = cfg.build_learner()
        optimizer = next(iter(learner._named_optimizers.values()))
        episodes = runner.sample()

        def fail_step(*args, **kwargs):
            raise RuntimeError("injected optimizer failure")

        monkeypatch.setattr(optimizer, "step", fail_step)
        with pytest.raises(RuntimeError, match="injected optimizer failure"):
            learner.update(episodes=episodes, num_epochs=1, minibatch_size=3)
        assert learner._checkpoint_updates == 0
        assert learner._checkpoint_learned == {}
        learner.after_gradient_based_update(timesteps={})
        assert (
            learner.metrics.peek(
                ("__all_modules__", "actual_optimizer_updates_this_update"), default=0
            )
            == 0
        )
        assert (
            learner.metrics.peek(
                ("__all_modules__", "actual_optimizer_minibatches_this_update"), default=0
            )
            == 0
        )
    finally:
        runner.stop()


def test_recurrent_minibatch_cyclic_wrap_keeps_unique_rows_and_synchronizes_modules():
    from ray.rllib.core.columns import Columns
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
    from support.bootstrap import ACT_SPACE, OBS_SPACE, learner_config, mapping
    from support.checkpoints import ProbePPOModule

    from common.runtime_log import runtime_print

    cfg = learner_config(stateful=True, shared=False).training(use_kl_loss=False)
    spec = cfg.rl_module_spec
    spec.module_class = ProbePPOModule
    cfg.rl_module(rl_module_spec=spec)
    learner = cfg.build_learner()
    episodes = []
    # p has 3 retained sequences (3, 3, 1 valid rows); q has one (1 valid).
    # The installed cyclic iterator selects 2 p sequences and 3 q sequences
    # per minibatch, continuing until the slower p module covers its epoch.
    for agent, values in (("a", [3, 7, 11, 13]), ("a", [20, 23, 27, 29, 31]), ("b", [40, 43])):
        ep = MultiAgentEpisode(
            observation_space=gym.spaces.Dict({agent: OBS_SPACE}),
            action_space=gym.spaces.Dict({agent: ACT_SPACE}),
            agent_to_module_mapping_fn=mapping,
        )
        module = learner.module[mapping(agent)]
        ep.add_env_reset(observations={agent: np.array([values[0]], np.float32)}, infos={})
        for index, value in enumerate(values[1:]):
            ep.add_env_step(
                observations={agent: np.array([value], np.float32)},
                actions={agent: 0},
                rewards={agent: 1.0},
                infos={},
                truncateds={"__all__": index == len(values) - 2},
                extra_model_outputs={
                    agent: {
                        Columns.STATE_OUT: module.get_initial_state(),
                        Columns.ACTION_DIST_INPUTS: np.zeros(2, np.float32),
                        Columns.ACTION_LOGP: np.float32(-np.log(2)),
                    }
                },
            )
        episodes.append(ep.to_numpy())
    p_episodes = copy.deepcopy(episodes[:2])
    prepared = {}
    original_prepare = learner._make_batch_if_necessary

    def record_prepared(training_data):
        batch = original_prepare(training_data)
        for module_id, data in batch.policy_batches.items():
            mask = data[Columns.LOSS_MASK]
            prepared[module_id] = data[Columns.OBS].squeeze(-1)[mask].tolist()
            assert mask.any(dim=1).all()
            assert data[Columns.STATE_IN]["h"].shape[0] == mask.shape[0]
        return batch

    learner._make_batch_if_necessary = record_prepared
    optimized = {"p": [], "q": []}
    original_loss = learner.compute_loss_for_module

    def record_loss(**kwargs):
        data = kwargs["batch"]
        mask = data[Columns.LOSS_MASK]
        optimized[kwargs["module_id"]].extend(data[Columns.OBS].squeeze(-1)[mask].tolist())
        assert mask.any()
        loss = original_loss(**kwargs)
        assert torch.isfinite(loss)
        return loss

    learner.compute_loss_for_module = record_loss
    result = learner.update(
        episodes=episodes, num_epochs=1, minibatch_size=6, shuffle_batch_per_epoch=False
    )
    assert prepared == {"p": [3, 7, 11, 20, 23, 27, 29], "q": [40]}
    assert optimized == {"p": [3, 7, 11, 20, 23, 27, 29, 3, 7, 11], "q": [40] * 6}
    for module_id, module in learner.module.items():
        assert int(module.value_normalizer.count) == 1
        assert all(torch.isfinite(value).all() for value in module.state_dict().values())
        assert result[module_id]["actual_valid_optimizer_rows_this_update"].peek() == len(
            optimized[module_id]
        )
        assert result[module_id]["actual_optimizer_minibatches_this_update"].peek() == 2
    for optimizer in learner._named_optimizers.values():
        for slots in optimizer.state.values():
            assert slots["step"].item() == 2
            assert all(torch.isfinite(value).all() for value in slots.values())
    assert result["__all_modules__"]["actual_valid_optimizer_rows_this_update"].peek() == 16
    assert result["__all_modules__"]["actual_optimizer_minibatches_this_update"].peek() == 2
    assert result["__all_modules__"]["actual_optimizer_updates_this_update"].peek() == 1
    runtime_print(
        "AuditA",
        "cyclic optimizer work",
        prepared=prepared,
        optimized=optimized,
        minibatches=2,
        updates=1,
    )

    q_before = copy.deepcopy(learner.module["q"].state_dict())
    second = learner.update(
        episodes=p_episodes, num_epochs=1, minibatch_size=6, shuffle_batch_per_epoch=False
    )
    for name in (
        "actual_valid_optimizer_rows_this_update",
        "actual_optimizer_minibatches_this_update",
        "actual_optimizer_updates_this_update",
    ):
        assert second["q"][name].peek() == 0
    assert second["p"]["actual_valid_optimizer_rows_this_update"].peek() == 10
    assert second["__all_modules__"]["actual_valid_optimizer_rows_this_update"].peek() == 10
    assert second["__all_modules__"]["actual_optimizer_minibatches_this_update"].peek() == 2
    assert second["__all_modules__"]["actual_optimizer_updates_this_update"].peek() == 1
    for key, value in q_before.items():
        torch.testing.assert_close(learner.module["q"].state_dict()[key], value, rtol=0, atol=0)
    runtime_print(
        "AuditA",
        "nonparticipating module work",
        p_valid=10,
        q_valid=0,
        q_minibatches=0,
        q_updates=0,
    )


@pytest.mark.parametrize("kind", ["sync", "sample", "non_none", "error", "remaining"])
def test_administrative_sync_barrier_never_discards_sample_results(kind):
    from training.checkpoint_boundary import _drain_runner_sync

    class Result:
        ok = kind != "error"

        def get(self):
            if kind == "error":
                return RuntimeError("actor failed")
            return ["trajectory"] if kind == "non_none" else None

    class Manager:
        pending = 1
        fetches = 0

        def num_outstanding_async_reqs(self, tag=None):
            return 0 if tag == "set_state" and kind == "sample" else self.pending

        def fetch_ready_async_reqs(self, *, tags, timeout_seconds):
            assert tags == "set_state" and timeout_seconds is None
            self.fetches += 1
            if kind != "remaining":
                self.pending = 0
            return [Result()]

    manager = Manager()
    if kind == "sync":
        _drain_runner_sync(manager)
        assert manager.pending == 0
    else:
        with pytest.raises((ValueError, RuntimeError)):
            _drain_runner_sync(manager)
    if kind == "sample":
        assert manager.fetches == 0 and manager.pending == 1


def test_actual_optimizer_lifetime_survives_restore():
    cfg, runner = make_runner(3, stateful=True, worker_index=1)
    try:
        learner = cfg.build_learner()
        for _ in range(2):
            learner.update(episodes=runner.sample(), num_epochs=2, minibatch_size=3)
        saved = learner.get_state()
        work = saved["atc_effective_state"].get("optimizer_work")
        assert work is not None, "true lifetime optimizer work must be checkpointed"
        assert work["__all_modules__"] == {"valid_rows": 12, "minibatches": 4, "updates": 2}
        restored = cfg.build_learner()
        restored.set_state(saved)
        assert restored.get_state()["atc_effective_state"]["optimizer_work"] == work
    finally:
        runner.stop()


def test_phase_moving_inference_device_drops_only_runner_cuda_rng():
    from training.checkpoint_boundary import _restore_runner, capture_process_rng
    from training.phase import PhaseTransition

    class Runner:
        def checkpoint_identity(self):
            return "training/1"

        def restore_boundary_state(self, state, *, validate_only, transition):
            pass

    random.seed(7)
    rng = capture_process_rng()
    expected = random.random()
    # One more CUDA state than this process has devices: the former inference GPU.
    rng["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)] * (torch.cuda.device_count() + 1)
    processes = {"training/1": {"runner": {}, "rng": rng}}
    stay = PhaseTransition("a", "b", {"lr": [1e-5, 2e-5]})
    move = PhaseTransition("a", "b", {"resources.inference_device": ["cuda", "cpu"]})
    for transition in (None, stay):
        error = _restore_runner(
            Runner(), processes=processes, validate_only=True, transition=transition
        )
        assert isinstance(error, ValueError)
    random.seed(0)
    assert (
        _restore_runner(Runner(), processes=processes, validate_only=False, transition=move)
        == "training/1"
    )
    assert random.random() == expected
