"""Owned Ray 2.53 PPO adapter for physical bootstrap and real-row GAE."""

import numpy as np
import ray
import torch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.postprocessing.zero_padding import (
    split_and_zero_pad_n_episodes,
    unpad_data_if_necessary,
)

from network.gae import compute_chunk_gae, normalize_valid_advantages

# Bump on any change that alters outputs; checkpoints record this token and reject mismatches.
CONNECTOR_VERSION = "ma-physical-bootstrap-gae-v3"

# Row order and masks are owned by this exact default multi-agent pipeline.
_EXPECTED_PIPELINE = (
    "AddOneTsToEpisodesAndTruncate",
    "AddObservationsFromEpisodesToBatch",
    "AddColumnsFromEpisodesToTrainBatch",
    "AddTimeDimToBatchAndZeroPad",
    "AddStatesFromEpisodesToBatch",
    "AgentToModuleMapping",
    "BatchIndividualItems",
    "NumpyToTensor",
    "GeneralAdvantageEstimation",
)


def install_gae_connector(pipeline, *, gamma, lambda_):
    """Replace exactly one stock GAE after parent build via supported pipeline APIs."""
    if ray.__version__ != "2.53.0":
        raise ValueError("owned PPO bootstrap supports only Ray 2.53.0")
    if (
        pipeline is None
        or tuple(type(c).__name__ for c in pipeline.connectors) != _EXPECTED_PIPELINE
    ):
        raise ValueError("unsupported PPO learner connector stack for owned bootstrap")
    if (
        any(not type(c).__module__.startswith("ray.rllib.connectors.") for c in pipeline.connectors)
        or type(pipeline.connectors[-1]) is not GeneralAdvantageEstimation
    ):
        raise ValueError("unsupported replacement in PPO learner connector stack")
    pipeline.insert_before(
        GeneralAdvantageEstimation, ATCGeneralAdvantageEstimation(gamma=gamma, lambda_=lambda_)
    )
    pipeline.remove(GeneralAdvantageEstimation)


class ATCGeneralAdvantageEstimation(ConnectorV2):
    """Keep stock episode mapping/padding; replace only target/advantage semantics."""

    def __init__(self, input_observation_space=None, input_action_space=None, *, gamma, lambda_):
        super().__init__(input_observation_space, input_action_space)
        if ray.__version__ != "2.53.0":
            raise ValueError("owned PPO bootstrap supports only Ray 2.53.0")
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, *, rl_module, episodes, batch, shared_data=None, **kwargs):
        if not shared_data or not shared_data.get("_added_loss_mask_for_valid_episode_ts"):
            raise ValueError("owned GAE requires AddOneTs episode extension and loss mask")
        sa_episodes = list(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        )
        row_mapping = shared_data.get("memorized_map_structure")
        if not isinstance(row_mapping, dict) or set(row_mapping) != set(batch):
            raise ValueError("owned GAE requires the multi-agent row mapping")
        for module_id, data in batch.items():
            module = rl_module[module_id]
            if not isinstance(module, ValueFunctionAPI):
                raise ValueError("owned PPO requires a ValueFunctionAPI module")
            module_chunks = [e for e in sa_episodes if e.module_id == module_id]
            indexed = {(e.multi_agent_episode_id, e.agent_id): e for e in module_chunks}
            keys = row_mapping[module_id]
            if (
                len(indexed) != len(module_chunks)
                or len(keys) != len(indexed)
                or set(keys) != set(indexed)
            ):
                raise ValueError("owned GAE row mapping disagrees with episode chunks")
            chunks = [indexed[key] for key in keys]
            lengths = [len(e) for e in chunks]
            if not lengths or any(n < 2 for n in lengths):
                raise ValueError("owned GAE requires nonempty real episode chunks plus final value")
            stateful = module.is_stateful()
            max_seq_len = module.model_config.get("max_seq_len") if stateful else None

            def repad(array):
                return (
                    np.stack(
                        split_and_zero_pad_n_episodes(
                            array, episode_lens=lengths, max_seq_len=max_seq_len
                        )
                    )
                    if stateful
                    else array
                )

            expected_mask = repad(
                np.concatenate([np.array([True] * (n - 1) + [False]) for n in lengths])
            )
            required = (Columns.LOSS_MASK, Columns.REWARDS, Columns.TERMINATEDS, Columns.TRUNCATEDS)
            if any(key not in data for key in required):
                raise ValueError("owned GAE missing required learner columns")
            arrays = {key: convert_to_numpy(data[key]) for key in required}
            if any(array.shape != expected_mask.shape for array in arrays.values()):
                raise ValueError("owned GAE column shape disagrees with episode lengths/padding")
            if arrays[Columns.LOSS_MASK].dtype != np.bool_ or not np.array_equal(
                arrays[Columns.LOSS_MASK], expected_mask
            ):
                raise ValueError("owned GAE loss mask disagrees with real episode rows")
            if (
                arrays[Columns.TERMINATEDS].dtype != np.bool_
                or arrays[Columns.TRUNCATEDS].dtype != np.bool_
            ):
                raise ValueError("owned GAE physical flags must be boolean")
            # The only value forward: includes the masked final observation, excludes
            # padding later. No installed helper or stock GAE runs on these values.
            with torch.no_grad():
                predictions = module.compute_values(data, embeddings=data.get(Columns.EMBEDDINGS))
            if (
                not isinstance(predictions, torch.Tensor)
                or tuple(predictions.shape) != expected_mask.shape
            ):
                raise ValueError("owned GAE value shape disagrees with final observation layout")
            values = unpad_data_if_necessary(lengths, convert_to_numpy(predictions))
            flat = {key: unpad_data_if_necessary(lengths, array) for key, array in arrays.items()}
            targets, advantages = np.zeros_like(values), np.zeros_like(values)
            offset = 0
            for chunk, length in zip(chunks, lengths, strict=True):
                stop = offset + length
                flags = flat[Columns.TERMINATEDS][offset:stop]
                if (
                    not flags[-1]
                    or flags[:-2].any()
                    or bool(flags[-2]) != bool(chunk.is_terminated)
                ):
                    raise ValueError("owned GAE unsupported physical/artificial termination layout")
                if len(chunk.observations) != length + 1:
                    raise ValueError("owned GAE missing final observation in extended episode")
                chunk_targets, chunk_advantages = compute_chunk_gae(
                    values[offset:stop],
                    flat[Columns.REWARDS][offset : stop - 1],
                    flags[:-1],
                    gamma=self.gamma,
                    lambda_=self.lambda_,
                )
                targets[offset : stop - 1], advantages[offset : stop - 1] = (
                    chunk_targets,
                    chunk_advantages,
                )
                offset = stop
            advantages = normalize_valid_advantages(advantages, flat[Columns.LOSS_MASK])
            data[Postprocessing.VALUE_TARGETS] = torch.as_tensor(
                repad(targets), device=predictions.device
            )
            data[Postprocessing.ADVANTAGES] = torch.as_tensor(
                repad(advantages), device=predictions.device
            )
        return batch
