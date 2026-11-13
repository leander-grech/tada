"""Effective state and CUDA moments survive the first restored optimizer update."""

import copy
import pickle

import numpy as np
import pytest
import torch
from ray.rllib.utils.numpy import convert_to_numpy
from support.bootstrap import learner_config

from training.learner_checkpoint import ATCPPOLearner


@pytest.mark.parametrize("capturable", [False, True])
def test_actual_cuda_adam_preserves_live_parameter_moments_and_steps(capturable):
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    parameter = torch.nn.Parameter(torch.ones(3, device="cuda"))
    optimizer = torch.optim.Adam(
        [parameter],
        lr=0.00073123456789,
        betas=(0.83123456789, 0.97123456789),
        eps=1.234567891e-8,
        weight_decay=0.01234567891,
        foreach=True,
        capturable=capturable,
    )
    parameter.grad = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    optimizer.step()
    saved = {
        "p_default_optimizer": {
            "module_id": "p",
            "state": copy.deepcopy(convert_to_numpy(optimizer.state_dict())),
        }
    }
    metadata = {k: copy.deepcopy(v) for k, v in optimizer.param_groups[0].items() if k != "params"}
    expected_avg = optimizer.state[parameter]["exp_avg"].clone()
    expected_sq = optimizer.state[parameter]["exp_avg_sq"].clone()
    learner = ATCPPOLearner.__new__(ATCPPOLearner)
    learner._named_optimizers = {"p_default_optimizer": optimizer}
    learner._device = torch.device("cuda")
    learner._set_optimizer_state(saved)
    assert {k: v for k, v in optimizer.param_groups[0].items() if k != "params"} == metadata
    assert parameter in optimizer.state
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], expected_avg)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], expected_sq)
    assert optimizer.state[parameter]["step"].item() == 1
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([3.0, 2.0, 1.0], device="cuda")
    optimizer.step()
    assert not torch.equal(before, parameter)
    assert optimizer.state[parameter]["step"].item() == 2
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg"],
        expected_avg * metadata["betas"][0]
        + (parameter.grad + before * metadata["weight_decay"]) * (1 - metadata["betas"][0]),
    )
    assert optimizer.param_groups[0]["capturable"] is capturable


def test_effective_state_roundtrip_and_phase_update_rebuild_real_schedulers():
    cfg = learner_config().training(use_kl_loss=False, lr=0.0017, entropy_coeff=0.031)
    learner = cfg.build_learner()
    first = learner.get_state()
    assert "atc_effective_state" in first
    assert learner.apply_phase_overrides({"learning_rate": 0.00037, "entropy_coeff": 0.007}) == {
        "learning_rate": 0.00037,
        "entropy_coeff": 0.007,
    }
    learner.after_gradient_based_update(timesteps={"num_env_steps_sampled_lifetime": 100})
    for optimizer, scheduler in learner._optimizer_lr_schedules.items():
        assert scheduler.get_current_value() == 0.00037
        assert optimizer.param_groups[0]["lr"] == 0.00037
    assert learner.entropy_coeff_schedulers_per_module["p"].get_current_value() == 0.007
    saved = learner.get_state()
    restored = cfg.build_learner()
    restored.set_state(saved)
    assert restored.get_state()["atc_effective_state"] == saved["atc_effective_state"]
    restored.after_gradient_based_update(timesteps={"num_env_steps_sampled_lifetime": 200})
    assert restored.entropy_coeff_schedulers_per_module["p"].get_current_value() == 0.007
    assert next(iter(restored._named_optimizers.values())).param_groups[0]["lr"] == 0.00037
    with pytest.raises(ValueError, match="sidecar"):
        restored.set_state({k: v for k, v in saved.items() if k != "atc_effective_state"})


def test_learner_state_roundtrip_restores_training_progress():
    from support.bootstrap import episode

    from training.checkpoint_boundary import RECEIPT_KEY, advance_learned_receipts

    cfg = learner_config().training(use_kl_loss=False)
    learner = cfg.build_learner()
    ep = episode()
    ep.custom_data[RECEIPT_KEY] = {
        "identity": "training/1",
        "sequence": 1,
        "env_steps": len(ep),
        "episode_id": ep.id_,
        "start": ep.env_t_started,
        "end": ep.env_t,
    }
    learner._checkpoint_learned = advance_learned_receipts({}, [ep])
    learner._checkpoint_updates = 3
    learner._checkpoint_optimizer_work = {"p": {"valid_rows": 11, "minibatches": 4, "updates": 3}}
    saved = learner.get_state()

    restored = cfg.build_learner()
    restored.set_state(copy.deepcopy(saved))

    assert restored._checkpoint_learned == learner._checkpoint_learned != {}
    assert restored._checkpoint_updates == 3
    assert restored._checkpoint_optimizer_work == learner._checkpoint_optimizer_work
    assert restored.get_state()["atc_effective_state"] == saved["atc_effective_state"]


@pytest.mark.parametrize(
    "bad",
    [
        {"lr": 0.1},
        {"learning_rate": 0},
        {"entropy_coeff": -1},
        {"entropy_coeff": float("nan")},
        {"learning_rate": True},
    ],
)
def test_phase_rejects_invalid_overrides_without_mutation(bad):
    learner = learner_config().training(use_kl_loss=False).build_learner()
    before = copy.deepcopy(learner.get_state()["atc_effective_state"])
    with pytest.raises(ValueError):
        learner.apply_phase_overrides(bad)
    assert learner.get_state()["atc_effective_state"] == before


@pytest.mark.parametrize(
    "setting",
    [
        {"use_kl_loss": True},
        {"lr": [[0, 0.1], [10, 0.01]]},
        {"entropy_coeff": [[0, 0.1], [10, 0.01]]},
    ],
)
def test_checkpoint_rejects_unsupported_schedules_but_allows_module_sync(setting):
    learner = learner_config().training(use_kl_loss=False).training(**setting).build_learner()
    partial = learner.get_state(components="rl_module")
    assert "atc_effective_state" not in partial
    learner.set_state(partial)
    with pytest.raises(ValueError, match="constant|KL"):
        learner.get_state()


def test_cuda_rng_roundtrip_uses_actor_device_generator():
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    from training.checkpoint_process import capture_process_rng, restore_process_rng

    torch.rand(1, device="cuda")
    state = capture_process_rng()
    expected = torch.rand(8, device="cuda")
    restore_process_rng(state)
    torch.testing.assert_close(torch.rand(8, device="cuda"), expected, rtol=0, atol=0)


def test_failed_learner_update_never_acknowledges_trajectory(monkeypatch):
    from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
    from support.bootstrap import episode

    from training.checkpoint_boundary import RECEIPT_KEY

    learner = learner_config().training(use_kl_loss=False).build_learner()
    ep = episode()
    ep.custom_data[RECEIPT_KEY] = {
        "identity": "training/1",
        "sequence": 1,
        "env_steps": len(ep),
        "episode_id": ep.id_,
        "start": ep.env_t_started,
        "end": ep.env_t,
    }

    def fail(*args, **kwargs):
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(PPOTorchLearner, "update", fail)
    with pytest.raises(RuntimeError, match="injected"):
        learner.update(episodes=[ep])
    assert learner._checkpoint_learned == {} and learner._checkpoint_updates == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_amsgrad",
        "extra_amsgrad",
        "shape",
        "dtype",
        "nonfinite_moment",
        "negative_second",
        "negative_max",
        "negative_step",
        "nonfinite_step",
        "overflow_step",
        "vector_step",
    ],
)
def test_malformed_adam_rejected_before_any_learner_state_changes(corruption):
    learner = learner_config().training(use_kl_loss=False).build_learner()
    optimizer = next(iter(learner._named_optimizers.values()))
    for parameter in optimizer.param_groups[0]["params"]:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = copy.deepcopy(learner.get_state())
    bad = copy.deepcopy(before)
    state = next(iter(bad["optimizer"].values()))["state"]
    group = state["param_groups"][0]
    slots = next(iter(state["state"].values()))
    for key, value in slots.items():
        slots[key] = np.asarray(value, dtype=np.float32)
    bad["rl_module"]["p"]["offset"] += 10
    if corruption == "missing_amsgrad":
        group["amsgrad"] = True
    elif corruption == "extra_amsgrad":
        slots["max_exp_avg_sq"] = slots["exp_avg_sq"].copy()
    elif corruption == "shape":
        slots["exp_avg"] = slots["exp_avg"].reshape(-1)[:1]
    elif corruption == "dtype":
        slots["exp_avg"] = slots["exp_avg"].astype("int64")
    elif corruption == "nonfinite_moment":
        slots["exp_avg"].flat[0] = float("nan")
    elif corruption == "negative_second":
        slots["exp_avg_sq"].flat[0] = -1
    elif corruption == "negative_max":
        group["amsgrad"] = True
        slots["max_exp_avg_sq"] = -slots["exp_avg_sq"].copy()
    elif corruption == "negative_step":
        slots["step"] = slots["step"] * 0 - 1
    elif corruption == "overflow_step":
        slots["step"] = np.asarray(1e100, dtype=np.float64)
    elif corruption == "nonfinite_step":
        slots["step"] = slots["step"] * float("inf")
    else:
        slots["step"] = slots["step"].reshape(1)
    bad["atc_effective_state"]["updates"] = 99
    with pytest.raises(ValueError, match="Adam"):
        learner.set_state(bad)
    assert pickle.dumps(learner.get_state()) == pickle.dumps(before)


@pytest.mark.parametrize("amsgrad", [False, True])
def test_adam_restore_preserves_lazy_parameters_and_cpu_first_update(amsgrad):
    from network.optimizer_state import restore_adam

    initialized = torch.nn.Parameter(torch.ones(3))
    lazy = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([initialized, lazy], amsgrad=amsgrad)
    initialized.grad = torch.ones_like(initialized)
    optimizer.step()
    saved = copy.deepcopy(convert_to_numpy(optimizer.state_dict()))
    restore_adam(optimizer, saved, "cpu")
    assert lazy not in optimizer.state
    initialized.grad = torch.ones_like(initialized)
    lazy.grad = torch.ones_like(lazy)
    optimizer.step()
    assert optimizer.state[initialized]["step"].item() == 2
    assert optimizer.state[lazy]["step"].item() == 1
