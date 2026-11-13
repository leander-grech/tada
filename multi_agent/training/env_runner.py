"""Narrow pre-ingestion compatibility shim for Ray's multi-agent done flags."""

import inspect
from copy import deepcopy
from typing import Any, cast

import numpy as np
import ray
from ray.rllib.env.multi_agent_env_runner import ENV_STEP_FAILURE, MultiAgentEnvRunner


class ATCEnvRunner(MultiAgentEnvRunner):
    """Preserve physical truth when globally truncated agents enter stock episodes.

    The environment retains its explicit audit dictionaries. Only the copied
    representation passed to MultiAgentEpisode drops redundant individual
    truncations, avoiding Ray's all-agents-done physical termination promotion.
    """

    def __init__(self, config, **kwargs):
        if ray.__version__ != "2.53.0" or tuple(
            inspect.signature(MultiAgentEnvRunner._try_env_step).parameters
        ) != ("self", "actions"):
            raise ValueError("owned env runner requires Ray 2.53.0 _try_env_step(self, actions)")
        from atc_env.episode import validate_episode_adapter

        validate_episode_adapter(self, _OWNED_NEW_EPISODE)
        super().__init__(config, **kwargs)
        self._checkpoint_emitted = {"sequence": 0, "env_steps": 0}
        if self.env is not None and self.env.unwrapped.autoreset_mode != "next_step":
            raise ValueError("owned env runner requires next_step vector autoreset")

    def _new_episode(self, env_index, episodes=None, call_on_episode_created=True):
        from atc_env.episode import ATCEpisode

        episodes = episodes if episodes is not None else self._episodes
        assert self.env is not None and self.env.envs is not None
        env = self.env.envs[env_index].unwrapped
        episodes[env_index] = cast(Any, ATCEpisode)(
            observation_space={aid: env.get_observation_space(aid) for aid in env.possible_agents},
            action_space={aid: env.get_action_space(aid) for aid in env.possible_agents},
            agent_to_module_mapping_fn=self.config.policy_mapping_fn,
        )
        if call_on_episode_created:
            self._make_on_episode_callback("on_episode_created", env_index, cast(Any, episodes))

    def checkpoint_identity(self):
        role = (
            "operational_evaluation"
            if self.config.env_config.get("evaluation_mode")
            else "training"
        )
        return f"{role}/{self.worker_index or 0}"

    def sample(self, **kwargs):
        from training.checkpoint_boundary import RECEIPT_KEY

        episodes = super().sample(**kwargs)
        for episode in episodes:
            self._checkpoint_emitted["sequence"] += 1
            self._checkpoint_emitted["env_steps"] += len(episode)
            episode.custom_data[RECEIPT_KEY] = {
                "identity": self.checkpoint_identity(),
                "sequence": self._checkpoint_emitted["sequence"],
                "env_steps": len(episode),
                "episode_id": episode.id_,
                "start": episode.env_t_started,
                "end": episode.env_t,
            }
        return episodes

    def _validate_boundary_layout(self):
        from training.checkpoint_boundary import CheckpointBoundaryError

        expected = (
            (
                self._env_to_module,
                (
                    "AddObservationsFromEpisodesToBatch",
                    "AddTimeDimToBatchAndZeroPad",
                    "AddStatesFromEpisodesToBatch",
                    "AgentToModuleMapping",
                    "BatchIndividualItems",
                    "NumpyToTensor",
                ),
            ),
            (
                self._module_to_env,
                (
                    "GetActions",
                    "TensorToNumpy",
                    "UnBatchToIndividualItems",
                    "ModuleToAgentUnmapping",
                    "RemoveSingleTsTimeRankFromBatch",
                    "NormalizeAndClipActions",
                    "ListifyDataForVectorEnv",
                ),
            ),
        )
        for pipeline, names in expected:
            pipeline = cast(Any, pipeline)
            if (
                tuple(type(c).__name__ for c in pipeline.connectors) != names
                or any(
                    not type(c).__module__.startswith("ray.rllib.connectors.")
                    for c in pipeline.connectors
                )
                or pipeline.get_state() != {}
            ):
                raise CheckpointBoundaryError("unsupported checkpoint connector layout/state")
        if self.env is not None and (
            self.num_envs != 1
            or self.env.envs is None
            or self.env.unwrapped.autoreset_mode != "next_step"
        ):
            raise CheckpointBoundaryError("unsupported checkpoint vector layout")

    def capture_boundary_state(self):
        from training.checkpoint_boundary import CheckpointBoundaryError

        self._validate_boundary_layout()
        for episode in getattr(self, "_episodes", []):
            if episode is not None and (
                episode.is_reset or episode.env_t != 0 or episode.agent_episodes
            ):
                raise CheckpointBoundaryError("active episode/recurrence cannot be checkpointed")
        if self._cached_to_module:
            raise CheckpointBoundaryError("cached connector output is pending")
        if self._done_episodes_for_metrics or self._ongoing_episodes_for_metrics:
            raise CheckpointBoundaryError("pending metrics trajectories must be drained")
        shared = self._shared_data or {}
        if set(shared) - {
            "memorized_map_structure",
            "vector_env_episodes_map",
            "_added_single_ts_time_rank",
        } or shared.get("memorized_map_structure"):
            raise CheckpointBoundaryError("unsupported pending connector shared state")
        if (
            "_added_single_ts_time_rank" in shared
            and shared["_added_single_ts_time_rank"] is not True
        ):
            raise CheckpointBoundaryError("invalid recurrent connector rank hint")
        environment = None
        if self.env is not None:
            assert self.env.envs is not None
            environment = self.env.envs[0].unwrapped.capture_boundary_state()
        return {
            "version": 1,
            "identity": self.checkpoint_identity(),
            "emitted": deepcopy(self._checkpoint_emitted),
            "environment": environment,
            "metrics": self.metrics.get_state(),
            "connectors": {
                "env_to_module": self._env_to_module.get_state(),
                "module_to_env": self._module_to_env.get_state(),
            },
            "weights_seq_no": self._weights_seq_no,
        }

    def restore_boundary_state(self, state, *, validate_only=False, transition=None):
        from training.checkpoint_boundary import CheckpointBoundaryError

        self._validate_boundary_layout()
        from training.checkpoint_envelopes import validate_runner_envelope

        validate_runner_envelope(
            state, identity=self.checkpoint_identity(), has_environment=self.env is not None
        )
        if (
            getattr(self, "_episodes", None)
            or self._cached_to_module
            or not self._needs_initial_reset
        ):
            raise CheckpointBoundaryError("restore runner before first sampling/reset")
        if self.env is not None:
            assert self.env.envs is not None
            self.env.envs[0].unwrapped.restore_boundary_state(
                state["environment"], validate_only=validate_only, transition=transition
            )
        elif state["environment"] is not None:
            raise CheckpointBoundaryError("local facade cannot restore an environment")
        if not validate_only:
            self.metrics.set_state(state["metrics"])
            self._checkpoint_emitted = deepcopy(state["emitted"])
            self._weights_seq_no = state["weights_seq_no"]
            self._shared_data = None
            self._env_to_module.set_state(state["connectors"]["env_to_module"])
            self._module_to_env.set_state(state["connectors"]["module_to_env"])

    def activate_environments(self, semantic_digest):
        """Release lazy native generation after the driver's all-actor inspection."""
        if self.env is not None:
            if self.env.envs is None:
                raise ValueError("runtime activation requires synchronous vector environments")
            environments = [wrapped.unwrapped for wrapped in self.env.envs]
            if any(
                not env._runtime_activation_required
                or env.consumed_semantic_digest != semantic_digest
                for env in environments
            ):
                raise ValueError("runtime activation environment identity mismatch")
            for env in environments:
                env._runtime_activated = True

    def _try_env_step(self, actions):
        result = super()._try_env_step(actions)
        if isinstance(result, str) and result == ENV_STEP_FAILURE:
            return result
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("unsupported vector env step result")
        if any(
            not isinstance(column, (list, tuple, np.ndarray)) or len(column) != self.num_envs
            for column in result
        ):
            raise ValueError("unsupported vector env step column shape")
        observations, rewards, terminateds, truncateds, infos = result
        terms, truncs = [], []
        for index in range(self.num_envs):
            if any(not isinstance(column[index], dict) for column in result):
                raise ValueError("vector env step must contain agent dictionaries")
            term, trunc = dict(terminateds[index]), dict(truncateds[index])
            if (
                "__all__" not in term
                or "__all__" not in trunc
                or any(
                    not isinstance(v, (bool, np.bool_)) for v in (*term.values(), *trunc.values())
                )
            ):
                raise ValueError("vector env step requires explicit boolean global flags")
            episode = self._episodes[index]
            if episode is None:
                raise ValueError("owned env runner requires an initialized episode")
            if episode.is_reset:
                previous = episode.agent_episodes
                known = (
                    set(previous)
                    | set(actions[index])
                    | set(observations[index])
                    | (set(term) | set(trunc)) - {"__all__"}
                )
                done = {aid for aid, ep in previous.items() if ep.is_done}
                done.update(aid for aid in known if term.get(aid, False) or trunc.get(aid, False))
                if known and done >= known and not term["__all__"] and not trunc["__all__"]:
                    raise ValueError("unsupported collective individual-only episode ending")
                for aid in known:
                    if aid in previous and previous[aid].is_done:
                        continue
                    physical = term["__all__"] or term.get(aid, False)
                    if (
                        not physical
                        and (trunc["__all__"] or trunc.get(aid, False))
                        and (aid not in observations[index] or observations[index][aid] is None)
                    ):
                        raise ValueError("nonterminal truncation requires final observation")
            if trunc["__all__"] and not term["__all__"]:
                trunc = {
                    aid: value for aid, value in trunc.items() if aid == "__all__" or not value
                }
            terms.append(term)
            truncs.append(trunc)
        return observations, rewards, terms, truncs, infos


_OWNED_NEW_EPISODE = ATCEnvRunner._new_episode
