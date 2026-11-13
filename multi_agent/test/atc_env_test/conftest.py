import pytest
from ray.rllib.env.env_context import EnvContext

from atc_env.environment import ATCEnv

EVAL_SEEDS = [101, 202]


@pytest.fixture
def make_env():
    """Build a real evaluation-mode ATCEnv; override any env_config key by keyword.

    Environment-level constants (`Config.ACTION_SELECTION_MODE` etc.) must be patched
    before calling this, because the constructor reads them once.
    """

    def build(*, worker_index: int = 0, **config) -> ATCEnv:
        base = {"evaluation_mode": True, "evaluation_seeds": EVAL_SEEDS}
        return ATCEnv(EnvContext({**base, **config}, worker_index=worker_index))

    return build
