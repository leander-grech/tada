"""JAX network (jax_ppo/network.py) == the SB3/PyTorch policy, on real observations.

Run: TADA_SEQUENCE_OBS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python tests/test_jax_equivalence.py CKPT
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

from actions.action_set import sync_attention_to_checkpoint, sync_obs_width_to_checkpoint
from config.config import Config
from analysis.lookahead import env_config_for
from atc_env.windowed_env import WindowedATCEnv
from jax_ppo import network as jnet
from models.observations import AircraftScalarFeatures

jax.config.update("jax_default_matmul_precision", "highest")

if len(sys.argv) != 2:
    raise SystemExit("Provide a compatible checkpoint path")
CKPT = sys.argv[1]
sync_attention_to_checkpoint(CKPT); sync_obs_width_to_checkpoint(CKPT)
model = PPO.load(CKPT, device="cpu")
params = {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in model.policy.state_dict().items()}
UC = list(AircraftScalarFeatures.__dataclass_fields__).index("under_control")
HEADS = int(Config.ATTENTION_HEADS)

env = WindowedATCEnv({"evaluation_mode": True, **env_config_for(CKPT)})
obs_list = []
for seed in (41, 27911967):
    np.random.seed(seed); obs, _ = env.reset(seed=seed, options={"scenario_seed": seed})
    for _ in range(32):
        obs_list.append(obs)
        a, _ = model.predict(obs, deterministic=False)
        obs, r, te, tr, _ = env.step(a)
        if te or tr: break
batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

with torch.no_grad():
    t_obs = {k: torch.as_tensor(v) for k, v in batch.items()}
    ac_t, clr_t, again_t, v_t = model.policy._masked_heads(t_obs)
ac_j, clr_j, again_j, v_j = jax.jit(jnet.forward, static_argnums=(2, 3))(params, {k: jnp.asarray(v) for k, v in batch.items()}, HEADS, UC)

def maxdiff(a, b, mask=None):
    a, b = np.asarray(a), np.asarray(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.abs(a - b).max())

ac_valid = np.asarray(ac_t) > -1e7
clr_valid = np.asarray(clr_t) > -1e7
masks_match = (np.array_equal(ac_valid, np.asarray(ac_j) > -1e7)
               and np.array_equal(clr_valid, np.asarray(clr_j) > -1e7))
assert (again_t is None) == (again_j is None), "reselection head presence differs"
again_diff = 0.0
if again_t is not None:
    again_valid = np.asarray(again_t) > -1e7
    masks_match &= np.array_equal(again_valid, np.asarray(again_j) > -1e7)
    again_diff = maxdiff(again_t, again_j, again_valid)
print(f"{len(obs_list)} observations, obs width {batch['aircraft'].shape[-1]}")
print(f"aircraft logits  max |diff| {maxdiff(ac_t, ac_j, ac_valid):.2e}  (masking identical: {np.array_equal(ac_valid, np.asarray(ac_j) > -1e7)})")
print(f"clearance logits max |diff| {maxdiff(clr_t, clr_j, clr_valid):.2e}  (masking identical: {np.array_equal(clr_valid, np.asarray(clr_j) > -1e7)})")
if again_t is not None:
    print(f"again logits     max |diff| {again_diff:.2e}")
print(f"values           max |diff| {maxdiff(v_t, v_j):.2e}")

# log-probs and entropy of sampled actions
rng = np.random.default_rng(0)
acts = []
for i in range(len(obs_list)):
    ai = int(rng.choice(np.flatnonzero(ac_valid[i]))); ci = int(rng.choice(np.flatnonzero(clr_valid[i, ai])))
    action = (ai, ci)
    if again_t is not None:
        action += (int(rng.integers(2)) if again_valid[i, ai, ci] else 0,)
    acts.append(action)
acts = np.asarray(acts)
with torch.no_grad():
    _, lp_t, ent_t = model.policy.evaluate_actions(t_obs, torch.as_tensor(acts))
lp_j, ent_j = jnet.log_prob_entropy(ac_j, clr_j, jnp.asarray(acts), again_j)
print(f"log_prob         max |diff| {maxdiff(lp_t, lp_j):.2e}")
print(f"entropy          max |diff| {maxdiff(ent_t, ent_j):.2e}")
ok = max(maxdiff(ac_t, ac_j, ac_valid), maxdiff(clr_t, clr_j, clr_valid), maxdiff(v_t, v_j), maxdiff(lp_t, lp_j), maxdiff(ent_t, ent_j), again_diff) < 1e-4 and masks_match
print("EQUIVALENT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
