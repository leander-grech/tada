"""Behavior the ATC policy network must keep through refactors."""

import pytest
import torch
from ray.rllib.core.columns import Columns
from support.model import build_module, initial_state, obs_batch, step_slice

from actions.actions import Action
from config.config import Config

BATCH, STEPS = 3, 5


def test_stepwise_recurrent_inference_equals_one_sequence_pass():
    """Acting one step at a time with carried state matches training on the whole sequence."""
    module = build_module()
    obs = obs_batch(BATCH, STEPS)

    with torch.no_grad():
        sequence = module._forward_inference(
            {
                Columns.OBS: obs,
                Columns.STATE_IN: initial_state(module, BATCH),
                Columns.SEQ_LENS: torch.full((BATCH,), STEPS),
            }
        )
        state = initial_state(module, BATCH)
        stepwise_logits = []
        for t in range(STEPS):
            out = module._forward_inference(
                {Columns.OBS: step_slice(obs, t), Columns.STATE_IN: state}
            )
            state = out[Columns.STATE_OUT]
            stepwise_logits.append(out[Columns.ACTION_DIST_INPUTS].squeeze(1))

    stepwise = torch.stack(stepwise_logits, dim=1)
    sequence_logits = sequence[Columns.ACTION_DIST_INPUTS]
    assert stepwise.shape == sequence_logits.shape
    torch.testing.assert_close(stepwise, sequence_logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        state["gru_h"], sequence[Columns.STATE_OUT]["gru_h"], atol=1e-5, rtol=1e-4
    )


def test_recurrent_state_actually_changes_the_next_decision():
    """Guards the previous test against passing because the GRU is ignored."""
    module = build_module()
    obs = obs_batch(BATCH, 2)
    first = step_slice(obs, 0)
    second = step_slice(obs, 1)
    with torch.no_grad():
        carried = module._forward_inference(
            {Columns.OBS: first, Columns.STATE_IN: initial_state(module, BATCH)}
        )[Columns.STATE_OUT]
        with_history = module._forward_inference({Columns.OBS: second, Columns.STATE_IN: carried})
        fresh = module._forward_inference(
            {Columns.OBS: second, Columns.STATE_IN: initial_state(module, BATCH)}
        )
    assert not torch.allclose(
        with_history[Columns.ACTION_DIST_INPUTS], fresh[Columns.ACTION_DIST_INPUTS]
    )


@pytest.mark.parametrize("valid_waypoints", [1, 7, 19])
def test_padded_flight_plan_rows_do_not_affect_policy_or_value(valid_waypoints):
    """Whatever sits in masked waypoint rows must not leak into the decision."""
    module = build_module()
    garbage = obs_batch(BATCH, valid_waypoints=valid_waypoints)
    clean = obs_batch(BATCH, valid_waypoints=valid_waypoints)
    padding = clean["observations"]["flight_plan_mask"] == 0
    clean["observations"]["flight_plans_rel"][padding] = 0.0
    assert not torch.equal(
        garbage["observations"]["flight_plans_rel"], clean["observations"]["flight_plans_rel"]
    )

    with torch.no_grad():
        state = initial_state(module, BATCH)
        out_garbage = module._forward_train({Columns.OBS: garbage, Columns.STATE_IN: state})
        out_clean = module._forward_train({Columns.OBS: clean, Columns.STATE_IN: state})
        values_garbage = module.compute_values(
            {Columns.OBS: garbage, Columns.STATE_IN: state},
            embeddings=out_garbage[Columns.EMBEDDINGS],
        )
        values_clean = module.compute_values(
            {Columns.OBS: clean, Columns.STATE_IN: state},
            embeddings=out_clean[Columns.EMBEDDINGS],
        )

    torch.testing.assert_close(
        out_garbage[Columns.ACTION_DIST_INPUTS], out_clean[Columns.ACTION_DIST_INPUTS]
    )
    torch.testing.assert_close(values_garbage, values_clean)


def test_masked_actions_are_never_sampled():
    module = build_module()
    obs = obs_batch(BATCH)
    allowed = torch.zeros(BATCH, int(Config.NUM_ACTIONS), dtype=torch.bool)
    allowed[:, int(Action.DO_NOTHING)] = True
    allowed[0, int(Action.SLOW_DOWN_SMALL)] = True
    allowed[1, int(Action.SKIP_1_WAYPOINT)] = True
    obs["action_mask"] = allowed.float()

    with torch.no_grad():
        logits = module._forward_inference(
            {Columns.OBS: obs, Columns.STATE_IN: initial_state(module, BATCH)}
        )[Columns.ACTION_DIST_INPUTS].squeeze(1)

    draws = torch.distributions.Categorical(logits=logits).sample((500,))
    for row in range(BATCH):
        assert allowed[row, draws[:, row]].all()
    assert (logits[~allowed] < -1e30).all()
    assert (logits[allowed] > -1e30).all()


def test_all_zero_mask_row_falls_back_to_do_nothing():
    module = build_module()
    obs = obs_batch(BATCH)
    obs["action_mask"][1] = 0.0

    with torch.no_grad():
        logits = module._forward_inference(
            {Columns.OBS: obs, Columns.STATE_IN: initial_state(module, BATCH)}
        )[Columns.ACTION_DIST_INPUTS].squeeze(1)

    draws = torch.distributions.Categorical(logits=logits[1]).sample((200,))
    assert (draws == int(Action.DO_NOTHING)).all()


def test_mask_applies_per_time_step_in_sequences():
    module = build_module()
    obs = obs_batch(BATCH, STEPS)
    obs["action_mask"][:, 2:, int(Action.SLOW_DOWN_SMALL)] = 0.0

    with torch.no_grad():
        logits = module._forward_inference(
            {
                Columns.OBS: obs,
                Columns.STATE_IN: initial_state(module, BATCH),
                Columns.SEQ_LENS: torch.full((BATCH,), STEPS),
            }
        )[Columns.ACTION_DIST_INPUTS]

    slow = int(Action.SLOW_DOWN_SMALL)
    assert (logits[:, 2:, slow] < -1e30).all()
    assert (logits[:, :2, slow] > -1e30).all()
