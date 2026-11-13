"""Offline distillation must see each agent's decisions as the env runner did."""

import torch
from ray.rllib.core.columns import Columns
from support.model import build_module, initial_state, obs_batch, step_slice

from training.distill import agent_sequences, distill, module_batch, sequence_logits

STEPS = 6
LABEL = 1


def _world():
    """Two agents over one world; agent "b" appears only from step 2 and leaves after 4."""
    obs = obs_batch(2, STEPS)
    rows = {
        agent: [
            {
                "action_mask": step_slice(obs, t)["action_mask"][row].numpy(),
                "observations": {
                    k: v[row].numpy() for k, v in step_slice(obs, t)["observations"].items()
                },
            }
            for t in range(STEPS)
        ]
        for row, agent in enumerate("ab")
    }
    present = {"a": range(STEPS), "b": range(2, 5)}
    return [
        (float(t), {agent: (rows[agent][t], LABEL) for agent in "ab" if t in present[agent]})
        for t in range(STEPS)
    ], rows, present


def test_padded_agent_sequences_reproduce_stepwise_online_logits():
    module = build_module()
    steps, rows, present = _world()
    sequences = agent_sequences({7: steps})[7]
    batch, targets = module_batch(sequences, "cpu")
    with torch.no_grad():
        offline = sequence_logits(module, batch)
    for index, agent in enumerate("ab"):
        state = initial_state(module, 1)
        for position, t in enumerate(present[agent]):
            observation = rows[agent][t]
            single = {
                "action_mask": torch.from_numpy(observation["action_mask"])[None],
                "observations": {
                    k: torch.from_numpy(v)[None] for k, v in observation["observations"].items()
                },
            }
            with torch.no_grad():
                out = module._forward_inference({Columns.OBS: single, Columns.STATE_IN: state})
            state = out[Columns.STATE_OUT]
            torch.testing.assert_close(
                offline[index, position],
                out[Columns.ACTION_DIST_INPUTS].reshape(-1),
                atol=1e-5,
                rtol=1e-4,
            )
            assert targets[index, position] == LABEL
        assert (targets[index, len(present[agent]) :] < 0).all()


def test_distillation_learns_labels_that_differ_from_the_teacher():
    module = build_module()
    steps, _rows, _present = _world()
    sequences = agent_sequences({7: steps})[7]
    best, epoch, history = distill(
        module, sequences, sequences, epochs=40, lr=1e-2, batch_size=2, device="cpu", seed=0
    )
    assert history[0]["held_out"]["changed_rows"] > 0
    assert history[epoch]["held_out"]["changed_accuracy"] == 1.0
    module.load_state_dict(best)
    with torch.no_grad():
        logits = sequence_logits(module, module_batch(sequences, "cpu")[0])
    for index, (_, actions) in enumerate(sequences):
        assert logits[index, : len(actions)].argmax(-1).tolist() == actions.tolist()
