"""Unit tests for inference.engine behavior."""

# pylint: disable=protected-access

from types import SimpleNamespace
from typing import Any, cast

import pytest

import inference.engine as engine_module
from inference.engine import InferenceEngine


class _DummyBaseEnv:
    def __init__(self) -> None:
        from atc_env.completed_scenario import CompletedScenarioCollector

        self.completed_collector = CompletedScenarioCollector()
        self.max_steps_per_episode = 3
        self.received_worlds: list[object] = []
        self._action_history = [
            {"call_sign": "A", "action": 0, "x": 0.0, "y": 0.0, "step": 0, "reward": 0.0},
            {"call_sign": "B", "action": 3, "x": 1.0, "y": 2.0, "step": 1, "reward": 0.5},
        ]

    def set_next_world(self, world: object) -> None:
        self.received_worlds.append(world)


class _DummyEvalEnvRunner:
    def __init__(self) -> None:
        self.env = object()
        self.sample_calls: list[dict[str, object]] = []

    def sample(self, *, num_timesteps: int, explore: bool, force_reset: bool):
        from atc_env.completed_scenario import CompletedScenarioResult
        from outcomes.ledger import OutcomeSnapshot
        from outcomes.verification import classify_outcome

        env = engine_module.get_base_env(self.env)
        ledger = OutcomeSnapshot((), (), (), (), False)
        result = CompletedScenarioResult(
            getattr(env, "_next_attempt_id"),
            classify_outcome(ledger, None, threshold_s=60, policy_limit_reached=True),
            ledger,
            None,
            env._action_history,
            12,
            {"complete": True},
        )
        env.completed_collector.capture(result.attempt_id, result)
        self.sample_calls.append(
            {"num_timesteps": num_timesteps, "explore": explore, "force_reset": force_reset}
        )
        env._action_history = []
        return []


def test_run_episode_uses_input_world_without_preprocessing(monkeypatch):
    """run_episode should pass through the provided world unchanged."""

    base_env = _DummyBaseEnv()
    eval_runner = _DummyEvalEnvRunner()
    monkeypatch.setattr(engine_module, "get_base_env", lambda _env: base_env)

    engine = InferenceEngine.__new__(InferenceEngine)
    engine._algo = cast(Any, SimpleNamespace(eval_env_runner=eval_runner))  # pylint: disable=protected-access

    world = object()
    actions = engine.run_episode(world)

    assert base_env.received_worlds == [world]
    assert eval_runner.sample_calls == [
        {"num_timesteps": 1, "explore": False, "force_reset": False}
    ]
    assert len(actions) == 1 and actions[0]["action"] == 3
    assert base_env._action_history == []  # pylint: disable=protected-access


def test_run_episode_seeds_rngs_when_deterministic(monkeypatch):
    base_env = _DummyBaseEnv()
    eval_runner = _DummyEvalEnvRunner()
    monkeypatch.setattr(engine_module, "get_base_env", lambda _env: base_env)

    seed_calls: list[int] = []
    monkeypatch.setattr(
        InferenceEngine,
        "_seed_rngs",
        staticmethod(lambda seed=0: seed_calls.append(seed)),
    )

    engine = InferenceEngine.__new__(InferenceEngine)
    engine._deterministic = True  # pylint: disable=protected-access
    engine._deterministic_seed = 7  # pylint: disable=protected-access
    engine._algo = cast(Any, SimpleNamespace(eval_env_runner=eval_runner))  # pylint: disable=protected-access

    engine.run_episode(object())

    assert seed_calls == [7]


def test_run_episode_does_not_seed_rngs_when_non_deterministic(monkeypatch):
    base_env = _DummyBaseEnv()
    eval_runner = _DummyEvalEnvRunner()
    monkeypatch.setattr(engine_module, "get_base_env", lambda _env: base_env)

    seed_calls: list[int] = []
    monkeypatch.setattr(
        InferenceEngine,
        "_seed_rngs",
        staticmethod(lambda seed=0: seed_calls.append(seed)),
    )

    engine = InferenceEngine.__new__(InferenceEngine)
    engine._deterministic = False  # pylint: disable=protected-access
    engine._algo = cast(Any, SimpleNamespace(eval_env_runner=eval_runner))  # pylint: disable=protected-access

    engine.run_episode(object())

    assert not seed_calls


def test_configure_deterministic_inference_sets_seed_and_eval(monkeypatch):
    seed_calls: list[int] = []
    monkeypatch.setattr(
        InferenceEngine,
        "_seed_rngs",
        staticmethod(lambda seed=0: seed_calls.append(seed)),
    )
    deterministic_calls: list[bool] = []
    monkeypatch.setattr(
        InferenceEngine,
        "_configure_torch_determinism",
        staticmethod(lambda: deterministic_calls.append(True)),
    )

    module = SimpleNamespace(eval_calls=0)

    def _eval():
        module.eval_calls += 1

    module.eval = _eval

    engine = InferenceEngine.__new__(InferenceEngine)
    engine._deterministic_seed = 11  # pylint: disable=protected-access
    engine._algo = cast(Any, SimpleNamespace(get_module=lambda *_: module))  # pylint: disable=protected-access

    engine._configure_deterministic_inference()

    assert seed_calls == [11]
    assert deterministic_calls == [True]
    assert module.eval_calls == 1


def test_run_episode_requires_eval_env_runner():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine._algo = cast(Any, SimpleNamespace(eval_env_runner=None))  # pylint: disable=protected-access

    with pytest.raises(RuntimeError, match="Evaluation environment runner is not configured"):
        engine.run_episode(object())


@pytest.mark.parametrize("rejected", ["hazard", "incomplete"])
def test_engine_rejected_completed_receipt_preserves_evidence_and_cannot_serve(
    monkeypatch, rejected
):
    from dataclasses import replace

    from atc_env.completed_scenario import CompletedScenarioResult
    from inference.service import NoValidSolutionError, select_best_candidate
    from outcomes.evidence import ArrivalEvidence, ArrivalTargetSpec, TimingEvidence
    from outcomes.ledger import OutcomeSnapshot
    from outcomes.verification import VerificationResult, classify_outcome

    env = _DummyBaseEnv()
    runner = _DummyEvalEnvRunner()
    monkeypatch.setattr(engine_module, "get_base_env", lambda _: env)
    live = OutcomeSnapshot(
        (ArrivalTargetSpec("A", 0),),
        (ArrivalEvidence("A", True, True, TimingEvidence(4, 4, "exact")),),
        (),
        (),
        False,
    )
    verification = (
        VerificationResult(replace(live, failed=True), 1, False, "verification_hazard")
        if rejected == "hazard"
        else None
    )

    def sample(**kwargs):
        result = CompletedScenarioResult(
            getattr(env, "_next_attempt_id"),
            classify_outcome(live, None, threshold_s=60, policy_limit_reached=True),
            live,
            verification,
            (),
            0,
            {"complete": rejected != "incomplete"},
        )
        env.completed_collector.capture(result.attempt_id, result)

    monkeypatch.setattr(runner, "sample", sample)
    engine = InferenceEngine.__new__(InferenceEngine)
    engine._algo = cast(Any, SimpleNamespace(eval_env_runner=runner))
    result = engine.run_episode_with_metrics(object())
    assert result["end_reason"] == "unresolved"
    assert result["outcome_evidence"]["live_completed_count"] == 1
    assert result["total_deviation"] == 4
    with pytest.raises(NoValidSolutionError):
        select_best_candidate([cast(Any, {**result, "run_index": 0})], "engine")
