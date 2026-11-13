"""Pinned ATC episode adapter retaining authoritative global scenario flags."""

import ray
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

_BASE_SLOTS = (
    "id_",
    "agent_to_module_mapping_fn",
    "_agent_to_module_mapping",
    "observation_space",
    "action_space",
    "env_t_started",
    "env_t",
    "agent_t_started",
    "env_t_to_agent_t",
    "_hanging_actions_end",
    "_hanging_extra_model_outputs_end",
    "_hanging_rewards_end",
    "_hanging_rewards_begin",
    "is_terminated",
    "is_truncated",
    "agent_episodes",
    "_last_step_time",
    "_len_lookback_buffers",
    "_start_time",
    "_custom_data",
)


def validate_episode_adapter(runner=None, owned_new_episode=None):
    """Refuse an unsupported Ray version or slot layout and substituted owned hooks."""
    if ray.__version__ != "2.53.0" or MultiAgentEpisode.__slots__ != _BASE_SLOTS:
        raise ValueError("unsupported Ray ATC episode layout")
    for name, original in _OWNED_EPISODE_HOOKS.items():
        if ATCEpisode.__dict__.get(name) is not original:
            raise ValueError("substituted owned episode hook")
    if runner is not None:
        if not _owned_factory(getattr(runner._new_episode, "__func__", None), owned_new_episode):
            raise ValueError("substituted owned episode factory")
        if getattr(runner, "env", None) is not None:
            episodes = [None] * runner.num_envs
            for index in range(runner.num_envs):
                runner._new_episode(index, episodes=episodes, call_on_episode_created=False)
            if any(type(episode) is not ATCEpisode for episode in episodes):
                raise ValueError("actual factory returned an unsupported episode")


def _owned_factory(method, expected):
    # Ray wraps every actor method in a tracing closure over the owned function.
    # Accepting only single-cell closures relies on that wrapper shape in the pinned
    # Ray 2.53.0; revisit this unwrapping whenever Ray is upgraded.
    while method is not expected:
        match getattr(method, "__closure__", None):
            case (cell,):
                method = cell.cell_contents
            case _:
                return False
    return True


class ATCEpisode(MultiAgentEpisode):
    """Use simulator global truth during ingestion, retaining real agent terminals.

    Ray infers global completion from previously observed agents before ingesting
    new arrivals. ATC worlds can have future starts, so only the explicit global
    flag may close a world. The inherited slot remains the durable flag; the
    temporary guard exists only while the inherited method ingests one step.
    """

    __slots__ = ("_authoritative_termination",)

    @property
    def is_terminated(self):
        return MultiAgentEpisode.__dict__["is_terminated"].__get__(self)

    @is_terminated.setter
    def is_terminated(self, value):
        authoritative = getattr(self, "_authoritative_termination", None)
        MultiAgentEpisode.__dict__["is_terminated"].__set__(
            self, value if authoritative is None else authoritative
        )

    def add_env_step(
        self,
        observations,
        actions,
        rewards,
        infos=None,
        *,
        terminateds=None,
        truncateds=None,
        extra_model_outputs=None,
    ):
        if not isinstance(terminateds, dict) or "__all__" not in terminateds:
            raise ValueError("ATC episode requires authoritative global termination")
        self._authoritative_termination = bool(terminateds["__all__"])
        try:
            return super().add_env_step(
                observations,
                actions,
                rewards,
                infos,
                terminateds=terminateds,
                truncateds=truncateds,
                extra_model_outputs=extra_model_outputs,
            )
        finally:
            del self._authoritative_termination

    @classmethod
    def _from_base(cls, episode):
        """Transfer the complete pinned base layout without reconstructing data."""
        if type(episode) is not MultiAgentEpisode:
            raise ValueError("unsupported ATC episode recreation")
        result = cls.__new__(cls)
        for name in _BASE_SLOTS:
            descriptor = getattr(MultiAgentEpisode, name)
            try:
                value = descriptor.__get__(episode)
            except AttributeError:
                continue  # Ray leaves optional slots absent on empty episodes.
            descriptor.__set__(result, value)
        return result

    def cut(self, len_lookback_buffer=0):
        return self._from_base(super().cut(len_lookback_buffer))

    @staticmethod
    def from_state(state):
        return ATCEpisode._from_base(MultiAgentEpisode.from_state(state))


_OWNED_EPISODE_HOOKS = {
    name: ATCEpisode.__dict__[name]
    for name in ("is_terminated", "add_env_step", "cut", "from_state", "_from_base")
}
