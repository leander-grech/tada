"""Reselection (TADA_MAX_PICKS > 1): returns, masks, env equivalence, warm start, torch == JAX.

  1. GAE with a per-transition discount (jax_ppo.ppo.compute_gae): equal to SB3's
     RolloutBuffer when every discount is gamma, and to the brute-force lambda-return sum when
     some transitions are zero-time (discount 1).
  2. Picks mode with again=0 always reproduces the one-pick env exactly (same rewards, same
     observations apart from the added column/masks, same episode metrics).
  3. Random policy WITH zero-time picks: every mask invariant holds at every decision, the clock
     stands still on a zero-time pick and moves otherwise, and the potential-based shaping
     telescopes under the learner's per-transition discount,
         sum_t Gamma_t F_t = Gamma_T Phi(s_T) - Phi(s_0),   Gamma_t = prod_{k<t} gamma_k,
     so it cannot change the optimal policy. The same sum with gamma on the zero-time picks too
     is printed to show it does NOT telescope.
  4. A policy warm-started from a one-pick policy (fresh again head, zero column) gives
     exactly the source policy's aircraft / clearance logits and value on a step's first pick.
  5. The torch policy and jax_ppo/network.py agree on logits, values, log-probs and the exact
     entropy with the again head (including masked flags and a step's later picks).

Run: TADA_SEQUENCE_OBS=1 TADA_MAX_PICKS=2 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
         python tests/test_multi_pick.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

torch.set_num_threads(1)
import jax
import jax.numpy as jnp
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from actions.actions import Action
from atc_env.windowed_env import WindowedATCEnv
from config.config import Config
from jax_ppo import network as jnet
from jax_ppo.ppo import compute_gae
from models.observations import AircraftScalarFeatures
from network.autoregressive_policy import ATCAutoregressivePolicy, warm_start_policy

jax.config.update("jax_default_matmul_precision", "highest")
assert Config.MAX_PICKS > 1 and Config.SEQUENCE_OBS, "run with TADA_SEQUENCE_OBS=1 TADA_MAX_PICKS=2"
GAMMA, LAM = 0.998, 0.95
DN = int(Action.DO_NOTHING)
UC = list(AircraftScalarFeatures.__dataclass_fields__).index("under_control")
PICK_COL = int(Config.AIRCRAFT_SCALAR_LEN) - 1
SEEDS = (41, 27911967, 1159417075)
def run_checks():
    failures = []


    def check(name, ok, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)


    def set_mode(picks: int):
        """Switch the process between one-pick and reselection (config is read at env/obs build)."""
        Config.MAX_PICKS = picks
        Config.AIRCRAFT_SCALAR_LEN = 16 + (1 if picks > 1 else 0)


    # ----------------------------------------------------------------------------- 1. GAE
    print("1. GAE with a per-transition discount")
    rng = np.random.default_rng(0)
    T, N = 64, 3
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    starts = (rng.random((T, N)) < 0.08).astype(np.float32)
    starts[0] = 1.0
    last_values = rng.normal(size=N).astype(np.float32)
    last_starts = np.array([0.0, 1.0, 0.0], np.float32)

    rb = RolloutBuffer(T, spaces.Box(-1, 1, (1,)), spaces.Discrete(2), gamma=GAMMA, gae_lambda=LAM, n_envs=N)
    rb.rewards[:], rb.values[:], rb.episode_starts[:] = rewards, values, starts
    rb.compute_returns_and_advantage(torch.as_tensor(last_values), last_starts)
    adv = compute_gae(rewards, values, starts, last_values, last_starts, np.full((T, N), GAMMA), LAM)
    check("constant gamma == SB3 RolloutBuffer", np.array_equal(adv, rb.advantages),
          f"max |diff| {np.abs(adv - rb.advantages).max():.1e}")

    disc = np.where(rng.random((T, N)) < 0.3, 1.0, GAMMA)
    adv = compute_gae(rewards, values, starts, last_values, last_starts, disc, LAM)
    nnt = np.concatenate([1.0 - starts[1:], (1.0 - last_starts)[None]], 0)
    nv = np.concatenate([values[1:], last_values[None]], 0)
    delta = rewards + disc * nv * nnt - values
    brute = np.zeros_like(adv, dtype=np.float64)
    for t in range(T):
        w = np.ones(N)
        for l in range(t, T):
            brute[t] += w * delta[l]
            w = w * disc[l] * LAM * nnt[l]
    check("zero-time transitions == brute-force lambda-return sum", np.allclose(adv, brute, atol=1e-5),
          f"max |diff| {np.abs(adv - brute).max():.1e}")


    # ----------------------------------------------------------------------------- env helpers
    def random_action(obs, rng, p_again):
        if "mask_select" in obs:
            sel = np.flatnonzero(obs["mask_select"])
        else:
            under = np.flatnonzero((obs["aircraft"][:, UC] > 0.5) & (obs["mask_aircraft"] > 0))
            sel = under if under.size else np.flatnonzero(obs["mask_aircraft"])
        a = int(rng.choice(sel))
        valid = np.flatnonzero(obs["mask_action_per_ac"][a])
        c = DN if (rng.random() < 0.5 or valid.size == 1) else int(rng.choice(valid[valid != DN]))
        again = int(rng.random() < p_again)
        return a, c, again


    def run_episode(env, seed, p_again, three, max_decisions=3000):
        """Random policy; returns the per-decision record and the final info."""
        rng = np.random.default_rng(seed)
        np.random.seed(seed % 2**32)  # the observation frame, as score_windowed pins it
        obs, _ = env.reset(seed=seed, options={"scenario_seed": seed})
        rec = [{"obs": obs, "phi": sum(env._phi_prev), "t": float(env.simulator.world.age_in_seconds),
                "order": list(env._prev_aircraft_order)}]
        done, info = False, {}
        while not done and len(rec) < max_decisions:
            a, c, again = random_action(obs, rng, p_again)
            act = np.array([a, c, again] if three else [a, c], dtype=np.int64)
            picked_before = list(getattr(env, "_picked", []))
            obs, r, te, tr, info = env.step(act)
            done = te or tr
            rec.append({"obs": obs, "r": float(r), "parts": info["reward_parts"], "info": info, "act": (a, c, again),
                        "phi": sum(env._phi_prev), "t": float(env.simulator.world.age_in_seconds),
                        "picked_before": picked_before, "done": done, "order": list(env._prev_aircraft_order)})
        return rec, info


    # ----------------------------------------------------------------------------- 2. again=0 == one-pick env
    print("2. picks mode with again=0 reproduces the one-pick env")
    for seed in SEEDS[:2]:
        set_mode(1)
        env1 = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs"})
        rec1, info1 = run_episode(env1, seed, 0.0, three=False)
        set_mode(2)
        env2 = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs"})
        rec2, info2 = run_episode(env2, seed, 0.0, three=True)
        same_len = len(rec1) == len(rec2)
        r_diff = max(abs(x["r"] - y["r"]) for x, y in zip(rec1[1:], rec2[1:])) if same_len else float("inf")
        obs_same = same_len and all(
            all(np.array_equal(x["obs"][k], y["obs"][k][..., :PICK_COL] if k == "aircraft" else y["obs"][k])
                for k in x["obs"])
            and not y["obs"]["aircraft"][:, PICK_COL].any()
            for x, y in zip(rec1, rec2))
        m1, m2 = info1["episode_metrics"], info2["episode_metrics"]
        metrics_same = all(m1[k] == m2[k] for k in m1)
        check(f"seed {seed}: {len(rec1) - 1} steps, rewards / observations / metrics identical",
              same_len and r_diff == 0.0 and obs_same and metrics_same,
              f"len {len(rec1)} vs {len(rec2)}, max |r diff| {r_diff:.1e}, obs {obs_same}, metrics {metrics_same}")


    # ----------------------------------------------------------------------------- 3. masks, clock, telescoping
    print("3. zero-time picks: masks, clock and potential-based shaping")
    set_mode(2)
    env = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs"})
    for seed in SEEDS:
        rec, info = run_episode(env, seed, 0.6, three=True)
        n_zero = sum(1 for x in rec[1:] if x["info"]["zero_time"])
        bad = []
        step_clear = []  # real clearances within the current clock step
        for prev, x in zip(rec[:-1], rec[1:]):
            o, i = prev["obs"], x["info"]
            a, c, again = x["act"]
            cs = i["callsign"]
            picked = x["picked_before"]
            # masks as the policy saw them: no aircraft cleared at this step is selectable
            if any(prev["order"][k] in picked for k in np.flatnonzero(o["mask_select"])):
                bad.append("an aircraft cleared at this step was selectable")
            if o["mask_again"][0] and len(picked) + 1 >= env._max_picks:
                bad.append("mask_again set with no budget left")
            if i["zero_time"] and not (again and c != DN and o["mask_again"][0]):
                bad.append("zero-time pick the masks did not allow")
            if (again and c != DN and o["mask_again"][0] and o["mask_select"][a]) and not i["zero_time"]:
                bad.append("allowed again=1 ignored")
            if i["zero_time"] != (x["t"] == prev["t"]):
                bad.append("clock moved on a zero-time pick, or stood still on a clock step")
            if i["clearance"] != DN:
                if cs in step_clear:
                    bad.append(f"{cs} cleared twice in one step")
                step_clear.append(cs)
            if len(step_clear) > env._max_picks:
                bad.append("more clearances in a step than the budget")
            if i["zero_time"]:
                # the aircraft just cleared is now unselectable, DO_NOTHING-only, and flagged
                s = list(x["obs"]["mask_select"])
                slot = x["order"].index(cs)
                if s[slot] or x["obs"]["mask_action_per_ac"][slot].sum() != 1 or x["obs"]["aircraft"][slot, PICK_COL] != 1.0:
                    bad.append("cleared aircraft still selectable / clearable / unflagged")
            else:
                step_clear = []
                if x["obs"]["aircraft"][:, PICK_COL].any():
                    bad.append("cleared-at-this-step column not reset at the clock step")
        # telescoping: shaping F_t = deviation + conflict + goal parts in outcome_pbrs
        g_ok, g_wrong, acc_ok, acc_wrong = 1.0, 1.0, 0.0, 0.0
        for x in rec[1:]:
            p = x["parts"]
            f = p.deviation + p.conflict + p.goal
            acc_ok += g_ok * f
            acc_wrong += g_wrong * f
            g_ok *= 1.0 if x["info"]["zero_time"] else GAMMA
            g_wrong *= GAMMA
        target_ok = g_ok * rec[-1]["phi"] - rec[0]["phi"]
        target_wrong = g_wrong * rec[-1]["phi"] - rec[0]["phi"]
        resid_ok, resid_wrong = abs(acc_ok - target_ok), abs(acc_wrong - target_wrong)
        end = info["episode_metrics"]["end_reason"]
        check(f"seed {seed}: {len(rec) - 1} decisions ({n_zero} zero-time, end {end}): masks and clock",
              not bad and n_zero > 0, "; ".join(sorted(set(bad))[:4]))
        check(f"seed {seed}: shaping telescopes with discount 1 on zero-time picks",
              resid_ok < 1e-6 * max(1.0, abs(target_ok)),
              f"residual {resid_ok:.1e} (with gamma on those picks instead: {resid_wrong:.1e})")


    # ----------------------------------------------------------------------------- 4. warm start
    print("4. warm start from a one-pick policy")
    set_mode(1)
    src_env = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs"})
    src = ATCAutoregressivePolicy(src_env.observation_space, src_env.action_space, lambda _: 3e-5)
    set_mode(2)
    env = WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs"})
    dst = ATCAutoregressivePolicy(env.observation_space, env.action_space, lambda _: 3e-5)
    added = warm_start_policy(dst, src)
    check("again head added, obs layer widened 16 -> 17", any("again_head" in a for a in added)
          and any("ac_mlp.0.weight" in a and "17" in a for a in added), "; ".join(added))
    rec, _ = run_episode(env, SEEDS[2], 0.6, three=True, max_decisions=120)
    first = [x["obs"] for x in rec if not x["obs"]["aircraft"][:, PICK_COL].any()]
    batch = {k: np.stack([o[k] for o in first]) for k in first[0]}
    old = {k: v for k, v in batch.items() if k not in ("mask_select", "mask_again")}
    old["aircraft"] = old["aircraft"][..., :PICK_COL]
    with torch.no_grad():
        a0, c0, v0 = src._masked_logits({k: torch.as_tensor(v) for k, v in old.items()})
        a1, c1, v1 = dst._masked_logits({k: torch.as_tensor(v) for k, v in batch.items()})
    check(f"{len(first)} first-pick observations: identical logits and values",
          torch.equal(a0, a1) and torch.equal(c0, c1) and torch.equal(v0, v1),
          f"max |diff| {max((a0 - a1).abs().max(), (c0 - c1).abs().max(), (v0 - v1).abs().max()):.1e}")


    # ----------------------------------------------------------------------------- 5. torch == JAX
    print("5. torch policy == JAX network with the again head")
    with torch.no_grad():  # make the again head non-trivial so every term is exercised
        g = torch.Generator().manual_seed(0)
        dst.ar_net.again_head[2].weight.normal_(0.0, 0.3, generator=g)
        dst.ar_net.again_head[2].bias.normal_(0.0, 1.0, generator=g)
    obs_all = [x["obs"] for x in rec]
    batch = {k: np.stack([o[k] for o in obs_all]) for k in obs_all[0]}
    t_obs = {k: torch.as_tensor(v) for k, v in batch.items()}
    params = {k: jnp.asarray(v.detach().numpy()) for k, v in dst.state_dict().items()}
    with torch.no_grad():
        ac_t, clr_t, ag_t, v_t = dst._masked_heads(t_obs)
        acts_t, _, lp_fwd = dst.forward(t_obs, deterministic=False)
    ac_j, clr_j, ag_j, v_j = jax.jit(jnet.forward, static_argnums=(2, 3, 4))(
        params, {k: jnp.asarray(v) for k, v in batch.items()}, int(Config.ATTENTION_HEADS), UC, DN)


    def maxdiff(a, b, mask=None):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.abs(a[mask] - b[mask]).max()) if mask is not None else float(np.abs(a - b).max())


    ac_v, clr_v, ag_v = (np.asarray(x) > -1e7 for x in (ac_t, clr_t, ag_t))
    n_allowed = int(np.asarray(batch["mask_again"]).sum())
    print(f"  {len(obs_all)} observations, {n_allowed} with again allowed, "
          f"{int(np.asarray(batch['aircraft'])[:, :, PICK_COL].any(-1).sum())} later picks")
    check("masking identical (aircraft, clearance, again)",
          np.array_equal(ac_v, np.asarray(ac_j) > -1e7) and np.array_equal(clr_v, np.asarray(clr_j) > -1e7)
          and np.array_equal(ag_v, np.asarray(ag_j) > -1e7))
    check("again masked on DO_NOTHING and without budget",
          not ag_v[..., DN].any() and not ag_v[np.asarray(batch["mask_again"])[:, 0] == 0].any())
    d_logits = max(maxdiff(ac_t, ac_j, ac_v), maxdiff(clr_t, clr_j, clr_v), maxdiff(ag_t, ag_j, ag_v), maxdiff(v_t, v_j))
    check("logits and values", d_logits < 1e-4, f"max |diff| {d_logits:.1e}")
    acts = acts_t.numpy()
    with torch.no_grad():
        _, lp_t, ent_t = dst.evaluate_actions(t_obs, torch.as_tensor(acts))
    lp_j, ent_j = jnet.log_prob_entropy(ac_j, clr_j, jnp.asarray(acts), ag_j)
    check("forward log-prob == evaluate_actions log-prob (torch)", maxdiff(lp_fwd, lp_t) < 1e-5,
          f"max |diff| {maxdiff(lp_fwd, lp_t):.1e}")
    check("log-prob and exact entropy, torch == JAX", max(maxdiff(lp_t, lp_j), maxdiff(ent_t, ent_j)) < 1e-4,
          f"log-prob {maxdiff(lp_t, lp_j):.1e}, entropy {maxdiff(ent_t, ent_j):.1e}; "
          f"{int(acts[:, 2].sum())} sampled again=1")
    # the exact entropy against enumeration of the whole (aircraft, clearance, again) tree
    with torch.no_grad():
        pa = torch.softmax(ac_t, -1); pc = torch.softmax(clr_t, -1); pg = torch.sigmoid(ag_t)
        joint = torch.stack([pa[..., None] * pc * (1 - pg), pa[..., None] * pc * pg], -1)
        h_enum = -(joint * torch.log(joint.clamp_min(1e-30))).flatten(1).sum(-1)
    check("exact entropy == enumeration of the joint", maxdiff(ent_t, h_enum) < 1e-4,
          f"max |diff| {maxdiff(ent_t, h_enum):.1e}")

    print("\nALL PASS" if not failures else f"\nFAILED: {failures}")
    return failures


def test_multi_pick_contracts():
    failures = run_checks()
    assert not failures, failures


if __name__ == "__main__":
    sys.exit(0 if not run_checks() else 1)
