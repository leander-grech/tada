"""One PPO update in JAX (jax_ppo/ppo.py) == one stable_baselines3 PPO.train(), same data.

Both learners start from the same checkpoint and the same collected rollout, with the same
minibatch order. The parameters after one update (5 epochs) are compared, for a normal update
and for a critic warm-up update (only the value head may move).

Run: TADA_SEQUENCE_OBS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python tests/test_jax_ppo_update.py CKPT
With TADA_MAX_PICKS=2 the same comparison runs with reselection: the checkpoint is warm-started
into the three-part action policy and its again head given random output weights, so the flag's
log-probability and entropy terms are in the update.
"""
# These standalone numerical checks require an explicitly supplied checkpoint.
if __name__ != "__main__":
    import pytest
    pytest.skip("run this check as a script with a compatible checkpoint", allow_module_level=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import jax
import jax.numpy as jnp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from actions.action_set import load_policy_at_own_width, sync_attention_to_checkpoint, sync_obs_width_to_checkpoint
from actions.actions import Action
from config.config import Config
from analysis.lookahead import env_config_for
from atc_env.windowed_env import WindowedATCEnv
from jax_ppo.ppo import JaxPPO, PPOConfig
from models.observations import AircraftScalarFeatures
from network.autoregressive_policy import ATCAutoregressivePolicy, warm_start_policy

jax.config.update("jax_default_matmul_precision", "highest")
torch.manual_seed(0)
if len(sys.argv) != 2:
    raise SystemExit("Provide a compatible checkpoint path")
CKPT = sys.argv[1]
PICKS = int(Config.MAX_PICKS)
sync_attention_to_checkpoint(CKPT)
if PICKS == 1:
    sync_obs_width_to_checkpoint(CKPT)
LR, N_STEPS = 3e-5, 512

venv = DummyVecEnv([lambda: WindowedATCEnv({"evaluation_mode": False, "reward_mode": "outcome_pbrs", **env_config_for(CKPT)})])
model = PPO(ATCAutoregressivePolicy, venv, n_steps=N_STEPS, batch_size=256, n_epochs=5, gamma=0.998,
            gae_lambda=0.95, clip_range=0.2, ent_coef=0.003, vf_coef=0.25, max_grad_norm=1.5,
            target_kl=0.05, learning_rate=LR, device="cpu", seed=0)
src = load_policy_at_own_width(PPO, CKPT, device="cpu")
if PICKS == 1:
    model.policy.load_state_dict(src.policy.state_dict())
else:
    print("reselection:", "; ".join(warm_start_policy(model.policy, src.policy)))
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        model.policy.ar_net.again_head[2].weight.normal_(0.0, 0.3, generator=g)
        model.policy.ar_net.again_head[2].bias.normal_(0.0, 1.0, generator=g)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import configure_logger

model._logger = configure_logger(0, None, "", False)
model._setup_learn(N_STEPS, None, True, "t", False)
class _Noop(BaseCallback):
    def _on_step(self): return True
cb = _Noop(); cb.init_callback(model)
model.collect_rollouts(model.env, cb, model.rollout_buffer, N_STEPS)
rb = model.rollout_buffer

def sb3_rollout():
    flat = lambda x: np.asarray(x).reshape((N_STEPS,) + np.asarray(x).shape[2:])  # n_envs = 1
    return {"obs": {k: flat(v) for k, v in rb.observations.items()},
            "actions": flat(rb.actions).astype(np.int64), "old_log_prob": flat(rb.log_probs),
            "advantages": flat(rb.advantages), "returns": flat(rb.returns), "values": flat(rb.values)}

ROLLOUT = sb3_rollout()  # before SB3's train() flattens the buffer in place


def run(frozen: bool, target_kl: float = 0.05):
    start = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    model.target_kl = target_kl
    params0 = {k: jnp.asarray(v.numpy()) for k, v in start.items()}
    # SB3
    for name, p in model.policy.named_parameters():
        p.requires_grad = (not frozen) or name.startswith("ar_net.value_head.")
    model.policy.optimizer = torch.optim.Adam(
        [p for p in model.policy.parameters()], lr=LR, eps=1e-5)
    np.random.seed(123)
    model.train()
    sb3_after = {k: v.detach().numpy() for k, v in model.policy.state_dict().items()}
    # JAX, same start, same minibatch order
    cfg = PPOConfig(n_steps=N_STEPS, heads=int(Config.ATTENTION_HEADS), target_kl=target_kl,
                    under_control_col=list(AircraftScalarFeatures.__dataclass_fields__).index("under_control"),
                    do_nothing=int(Action.DO_NOTHING))
    agent = JaxPPO(params0, venv, cfg, lambda _: LR)
    agent.frozen = frozen
    class _NpShim:
        def permutation(self, n): return np.random.permutation(n)
    agent.np_rng = _NpShim()
    np.random.seed(123)
    stats = agent.train(ROLLOUT, 1.0)
    run.n_updates = stats["train/n_updates"]
    diffs = {k: float(np.abs(np.asarray(agent.params[k]) - sb3_after[k]).max()) for k in sb3_after}
    moved = {k: float(np.abs(sb3_after[k] - start[k].numpy()).max()) for k in sb3_after}
    model.policy.load_state_dict(start)
    for p in model.policy.parameters():
        p.requires_grad = True
    return diffs, moved

ok = True
# The third case stops early: a target KL small enough that the KL check fires inside the first
# epoch, so the minibatch that crosses it must be recorded but not applied, as SB3 does.
for frozen, tkl in ((False, 0.05), (True, 0.05), (False, 2e-6)):
    diffs, moved = run(frozen, tkl)
    worst = max(diffs, key=diffs.get)
    frozen_moved = max((moved[k] for k in moved if not k.startswith("ar_net.value_head.")), default=0.0)
    label = "critic warm-up" if frozen else ("normal update" if tkl == 0.05 else f"early stop (target_kl {tkl:g}, {run.n_updates:.0f} epoch(s))")
    print(f"{label}: SB3 moved params by up to {max(moved.values()):.2e}; "
          f"JAX vs SB3 after update: max |diff| {diffs[worst]:.2e} ({worst})"
          + (f"; non-value params moved {frozen_moved:.1e}" if frozen else ""))
    ok &= diffs[worst] < 1e-5 and (not frozen or frozen_moved == 0.0)
print("EQUIVALENT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
