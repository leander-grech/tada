"""The hard-solved objective (WindowedConfig.HARD_BRACKETS, main_jax.py --hard-bracket).

    python tests/test_hard_bracket.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.windowed_config import WindowedConfig  # noqa: E402
from atc_env.windowed_env import WindowedATCEnv, landing_value, landing_value_smooth  # noqa: E402

HARD = WindowedConfig.HARD_BRACKETS


def test_bracket_values():
    ladder = ((20.0, 7.0), (50.0, 4.0), (100.0, 1.0))
    assert landing_value(0.0, ladder) == 7.0
    assert landing_value(20.0, ladder) == 7.0
    assert landing_value(21.0, ladder) == 4.0
    assert landing_value(50.0, ladder) == 4.0
    assert landing_value(51.0, ladder) == 1.0
    assert landing_value(101.0, ladder) == WindowedConfig.LANDING_VALUE_BEYOND


def test_smooth_potential_is_monotone():
    xs = [float(x) for x in range(0, 700, 5)]
    for brackets in (None, HARD):
        v = [landing_value_smooth(x, brackets) for x in xs]
        assert all(a >= b for a, b in zip(v, v[1:]))


def test_env_penalty_follows_the_ladder():
    base = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs"})
    hard = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs", "landing_brackets": HARD})
    expected_delta = (HARD[0][1] - WindowedConfig.LANDING_BRACKETS[0][1]) * WindowedConfig.WINDOW_SIZE
    assert hard._pbrs_violation_penalty - base._pbrs_violation_penalty == expected_delta
    fixed = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs", "landing_brackets": HARD,
                            "pbrs_violation_penalty": 240.0})
    assert fixed._pbrs_violation_penalty == 240.0
    try:
        WindowedATCEnv({"evaluation_mode": True, "landing_brackets": [[60.0, 3.0], [30.0, 4.0]]})
    except AssertionError:
        pass
    else:
        raise AssertionError("a ladder with descending limits must be refused")
    # one step runs with the hard ladder
    obs, _ = hard.reset(seed=599310825, options={"scenario_seed": 599310825})
    hard.step(hard.action_space.sample() * 0)


def test_lookahead_reads_the_run_ladder():
    from analysis.lookahead import reward_config_for

    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / "run_meta.json").write_text(json.dumps({"windowed_config": {
            "reward_mode": "outcome_pbrs", "landing_brackets": [list(b) for b in HARD]}}))
        cfg = reward_config_for(str(run / "final_model.zip"))
        assert cfg["landing_brackets"] == tuple(tuple(b) for b in HARD)
        hash(tuple(sorted(cfg.items())))  # score_windowed puts it in a set


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
