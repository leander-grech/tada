"""Training redraws scenarios whose pre-autosolve deviation exceeds the cap; evaluation never does."""

from types import SimpleNamespace

from atc_env.environment import ATCEnv
from training.env import TrainingATCEnv


def _drive(monkeypatch, *, evaluation_mode: bool, cap: int, draws: list[dict[str, float]]):
    env = TrainingATCEnv.__new__(TrainingATCEnv)
    env.evaluation_mode = evaluation_mode
    env._max_initial_deviation_s = cap
    env._scenario_rejected = False
    remaining = iter(enumerate(draws))
    rejected_flags: list[bool] = []

    def fake_reset(self, *, seed=None, options=None):
        index, deviations = next(remaining)
        self._apply_initial_metric_baselines(
            SimpleNamespace(deviations_by_aircraft=deviations), None
        )
        self._journal_training_scenario("consumed")
        return index, {}

    monkeypatch.setattr(ATCEnv, "reset", fake_reset)
    monkeypatch.setattr(ATCEnv, "_apply_initial_metric_baselines", lambda *args: None)
    monkeypatch.setattr(
        TrainingATCEnv,
        "_journal_training_scenario",
        lambda self, status: rejected_flags.append(self._scenario_rejected),
    )
    observation, _ = env.reset()
    return observation, rejected_flags


DRAWS = [{"A": 100.0, "B": -1300.0}, {"A": 950.0}, {"A": -899.0, "B": 900.0}]


def test_training_redraws_until_every_flight_is_within_the_cap(monkeypatch):
    accepted, rejected = _drive(monkeypatch, evaluation_mode=False, cap=900, draws=DRAWS)
    assert (accepted, rejected) == (2, [True, True, False])


def test_disabled_cap_and_evaluation_keep_the_first_scenario(monkeypatch):
    for evaluation_mode, cap in ((False, 0), (True, 900)):
        accepted, rejected = _drive(
            monkeypatch, evaluation_mode=evaluation_mode, cap=cap, draws=DRAWS
        )
        assert (accepted, rejected) == (0, [False])
