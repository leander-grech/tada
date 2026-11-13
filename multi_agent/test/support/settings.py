"""Test settings that follow ``Config`` monkeypatching at every access."""

from atc_env.env_config import EnvRuntimeSettings, EnvSettings, resolve_env_settings
from config.settings import Settings


class LiveEnvSettings:
    """Stands in for ``ATCEnv.settings`` on ``__new__`` shells; re-resolves per access."""

    @property
    def core(self) -> Settings:
        return resolve_env_settings().core

    @property
    def env(self) -> EnvRuntimeSettings:
        return resolve_env_settings().env


def live_env_settings() -> EnvSettings:
    return LiveEnvSettings()  # ty: ignore[invalid-return-type]
