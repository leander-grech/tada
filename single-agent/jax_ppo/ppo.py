"""PPO in JAX, reproducing stable_baselines3.PPO with SB3-compatible configuration, step for step.

Only the learner is replaced. Environment stepping, reward normalisation and checkpoint files use
SB3's own env-side classes (SubprocVecEnv, VecNormalize, Monitor), and checkpoints are written
in SB3's format (jax_ppo/export.py), so every existing script loads a JAX-trained policy.

Matches SB3 2.x OnPolicyAlgorithm.collect_rollouts / RolloutBuffer / PPO.train:
  * truncated episodes bootstrap: reward += gamma * V(terminal_observation);
  * GAE with SB3's episode-start bookkeeping;
  * per-minibatch advantage normalisation with the UNBIASED std (torch.std), eps 1e-8;
  * clipped policy loss, value MSE (no value clipping), exact autoregressive entropy;
  * target-KL early stop checked BEFORE the optimiser step, at 1.5 * target_kl: the minibatch that
    crosses it is counted in the statistics but not applied, and the update ends there;
  * global-norm gradient clipping, Adam(eps=1e-5);
  * critic warm-up: only the value head trains; the other parameters get a FRESH Adam state
    when unfrozen -- in PyTorch frozen parameters accumulate no optimiser state at all.

One extension SB3 does not have: a per-transition discount. Under reselection
(Config.MAX_PICKS > 1) a pick that asks for another pick ("again") takes no simulated time, and
the env marks it with info["zero_time"]. That transition is discounted by 1, not gamma, in the
advantage recursion and in the truncation bootstrap -- the env's potential-based shaping uses the
same per-transition discount, so the shaping still telescopes (tests/test_multi_pick.py). Without
it every extra pick would cost (1 - gamma) * V(s'), a charge whose sign follows the value, which
the lexicographic objective does not contain. With no zero-time transitions this is exactly SB3.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_ppo import network as jnet


@dataclass
class PPOConfig:
    n_steps: int                 # per env
    batch_size: int = 256
    n_epochs: int = 5
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.25
    max_grad_norm: float = 1.5
    target_kl: Optional[float] = 0.05
    adam_eps: float = 1e-5
    heads: int = 4
    under_control_col: int = 11
    critic_warmup_steps: int = 0
    value_head_prefix: str = "ar_net.value_head."
    do_nothing: int = 0          # clearance index of DO_NOTHING (the again flag is masked on it)


def _to_jnp(obs: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in obs.items()}


def compute_gae(rewards, values, starts, last_values, last_starts, discounts, gae_lambda):
    """GAE with SB3's episode-start bookkeeping and a per-transition discount.

    All arrays are [T, N] except last_values / last_starts [N]. starts[t] = 1 where obs t begins
    an episode; discounts[t] is the discount of the transition t -> t+1 (gamma, or 1 for a
    zero-time pick). With discounts == gamma everywhere this is SB3's
    RolloutBuffer.compute_returns_and_advantage (tests/test_multi_pick.py). Returns advantages.
    """
    n_steps = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_values)
    # SB3 multiplies float32 arrays by Python floats: gamma, and gamma * lambda formed in double,
    # each rounded once to float32. Same here, per transition, so constant gamma is bit-exact.
    disc = np.asarray(discounts, np.float64)
    g32, gl32 = disc.astype(np.float32), (disc * float(gae_lambda)).astype(np.float32)
    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            nnt, nv = 1.0 - last_starts, last_values
        else:
            nnt, nv = 1.0 - starts[step + 1], values[step + 1]
        delta = rewards[step] + g32[step] * nv * nnt - values[step]
        last = delta + gl32[step] * nnt * last
        adv[step] = last
    return adv


class JaxPPO:
    def __init__(self, params: Dict[str, jnp.ndarray], venv, cfg: PPOConfig,
                 lr_schedule: Callable[[float], float], seed: int = 0):
        self.params = params
        self.venv = venv
        self.cfg = cfg
        self.lr_schedule = lr_schedule
        self.key = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)
        self.num_timesteps = 0
        self.n_updates = 0
        self.frozen = False
        self.labels = {k: ("value" if k.startswith(cfg.value_head_prefix) else "rest") for k in params}

        # Adam's direction per parameter group (scale_by_adam = torch.optim.Adam's
        # m_hat / (sqrt(v_hat) + eps)), with the learning rate applied in apply(). Two groups so
        # the critic warm-up can hand the unfrozen parameters a fresh Adam state.
        adam = lambda: optax.scale_by_adam(eps=cfg.adam_eps)
        self.tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.multi_transform({"value": adam(), "rest": adam()}, self.labels),
        )
        self.opt_state = self.tx.init(params)
        self._build_jit()

    # ------------------------------------------------------------------ jitted pieces
    def _build_jit(self):
        cfg = self.cfg
        heads, ucol, dn = cfg.heads, cfg.under_control_col, cfg.do_nothing

        def act(params, obs, key):
            ac, clr, again, v = jnet.forward(params, obs, heads, ucol, dn)
            actions, logp = jnet.sample(key, ac, clr, again)
            return actions, v, logp

        def value(params, obs):
            return jnet.forward(params, obs, heads, ucol, dn)[3]

        def loss_fn(params, mb, clip_range):
            ac, clr, again, v = jnet.forward(params, mb["obs"], heads, ucol, dn)
            logp, ent = jnet.log_prob_entropy(ac, clr, mb["actions"], again)
            adv = mb["advantages"]
            adv = (adv - adv.mean()) / (adv.std(ddof=1) + 1e-8)
            log_ratio = logp - mb["old_log_prob"]
            ratio = jnp.exp(log_ratio)
            pg_loss = -jnp.minimum(adv * ratio, adv * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)).mean()
            v_loss = ((mb["returns"] - v) ** 2).mean()
            ent_loss = -ent.mean()
            loss = pg_loss + cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss
            aux = dict(
                pg_loss=pg_loss, v_loss=v_loss, ent_loss=ent_loss, loss=loss,
                clip_frac=(jnp.abs(ratio - 1) > clip_range).astype(jnp.float32).mean(),
                approx_kl=jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean()),
            )
            return loss, aux

        def grad_step(params, mb, clip_range, frozen):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, clip_range)
            # Critic warm-up: everything but the value head gets no gradient (torch: requires_grad
            # False), so it is excluded from the global clipping norm as well.
            grads = {k: (g if (not frozen or k.startswith(cfg.value_head_prefix)) else jnp.zeros_like(g))
                     for k, g in grads.items()}
            return grads, aux

        def apply(params, opt_state, grads, lr):
            direction, opt_state = self.tx.update(grads, opt_state, params)
            return jax.tree_util.tree_map(lambda p, d: p - lr * d, params, direction), opt_state

        kl_stop = None if cfg.target_kl is None else 1.5 * cfg.target_kl

        def epoch(params, opt_state, data, perm_mb, lr, clip_range, frozen):
            """One epoch as ONE compiled call: a scan over the minibatches (perm_mb [n_mb, bs]).

            SB3 semantics, step for step: a minibatch's statistics are recorded, then the KL check,
            then the optimiser step. Once the check fires the rest of the epoch is computed but
            neither recorded nor applied (`processed`, `stopped`), and train() runs no further epoch.
            One call per epoch instead of two per minibatch: on a host with slow cores the per-call
            Python and dispatch overhead, not the GPU, set the update time.
            """
            def body(carry, idx):
                params, opt_state, stopped = carry
                mb = jax.tree_util.tree_map(lambda x: x[idx], data)
                grads, aux = grad_step(params, mb, clip_range, frozen)
                processed = jnp.logical_not(stopped)
                if kl_stop is not None:
                    stopped = stopped | (aux["approx_kl"] > kl_stop)
                new_params, new_opt = apply(params, opt_state, grads, lr)
                keep = lambda new, old: jnp.where(stopped, old, new)
                params = jax.tree_util.tree_map(keep, new_params, params)
                opt_state = jax.tree_util.tree_map(keep, new_opt, opt_state)
                return (params, opt_state, stopped), (aux, processed)

            (params, opt_state, stopped), (auxs, processed) = jax.lax.scan(
                body, (params, opt_state, jnp.asarray(False)), perm_mb)
            return params, opt_state, stopped, auxs, processed

        self._act = jax.jit(act)
        self._value = jax.jit(value)
        self._epoch = jax.jit(epoch, static_argnums=(6,))

    def _reset_rest_optimizer(self):
        """Fresh Adam state for the non-value parameters (the unfreeze of the critic warm-up)."""
        fresh = self.tx.init(self.params)
        clip_state, inner = self.opt_state
        states = dict(inner.inner_states)
        states["rest"] = fresh[1].inner_states["rest"]
        self.opt_state = (clip_state, inner._replace(inner_states=states))

    def set_frozen(self, frozen: bool):
        if self.frozen and not frozen:
            self._reset_rest_optimizer()
        self.frozen = frozen

    # ------------------------------------------------------------------ rollouts
    def collect(self, obs, episode_starts, on_step: Callable[[Dict[str, Any]], None]):
        cfg, n_envs = self.cfg, self.venv.num_envs
        buf = {k: np.zeros((cfg.n_steps, n_envs) + v.shape[1:], v.dtype) for k, v in obs.items()}
        actions = np.zeros((cfg.n_steps, n_envs, len(self.venv.action_space.nvec)), np.int64)
        rewards = np.zeros((cfg.n_steps, n_envs), np.float32)
        starts = np.zeros((cfg.n_steps, n_envs), np.float32)
        discounts = np.full((cfg.n_steps, n_envs), cfg.gamma, np.float64)
        values = np.zeros((cfg.n_steps, n_envs), np.float32)
        logps = np.zeros((cfg.n_steps, n_envs), np.float32)
        t_env = 0.0
        for step in range(cfg.n_steps):
            self.key, sub = jax.random.split(self.key)
            a, v, lp = self._act(self.params, _to_jnp(obs), sub)
            a = np.asarray(a)
            t0 = time.perf_counter()
            new_obs, r, dones, infos = self.venv.step(a)
            t_env += time.perf_counter() - t0
            self.num_timesteps += n_envs
            r = np.asarray(r, np.float32).copy()
            for i, info in enumerate(infos):
                if info.get("zero_time", False):
                    discounts[step, i] = 1.0
            for i, done in enumerate(dones):
                if done and infos[i].get("terminal_observation") is not None and infos[i].get("TimeLimit.truncated", False):
                    term = {k: np.asarray(x)[None] for k, x in infos[i]["terminal_observation"].items()}
                    r[i] += discounts[step, i] * float(np.asarray(self._value(self.params, _to_jnp(term)))[0])
            for k in buf:
                buf[k][step] = obs[k]
            actions[step], rewards[step], starts[step] = a, r, episode_starts
            values[step], logps[step] = np.asarray(v), np.asarray(lp)
            on_step({"infos": infos, "dones": dones, "rewards": r})
            obs, episode_starts = new_obs, dones.astype(np.float32)
        last_values = np.asarray(self._value(self.params, _to_jnp(obs)))
        adv = compute_gae(rewards, values, starts, last_values, episode_starts, discounts, cfg.gae_lambda)
        flat = lambda x: x.reshape((cfg.n_steps * n_envs,) + x.shape[2:])
        rollout = {
            "obs": {k: flat(v) for k, v in buf.items()},
            "actions": flat(actions), "old_log_prob": flat(logps),
            "advantages": flat(adv), "returns": flat(adv + values), "values": flat(values),
        }
        return rollout, obs, episode_starts, t_env

    # ------------------------------------------------------------------ update
    def train(self, rollout, progress_remaining: float) -> Dict[str, float]:
        cfg = self.cfg
        lr = float(self.lr_schedule(progress_remaining))
        data = jax.tree_util.tree_map(jnp.asarray, {k: v for k, v in rollout.items() if k != "values"})
        n = rollout["actions"].shape[0]
        n_mb = n // cfg.batch_size
        if n_mb * cfg.batch_size != n:
            raise ValueError(f"rollout size {n} must be a multiple of batch_size {cfg.batch_size}")
        stats = {k: [] for k in ("pg_loss", "v_loss", "ent_loss", "clip_frac")}
        approx_kls, last_loss = [], float("nan")
        for _ in range(cfg.n_epochs):
            perm = self.np_rng.permutation(n).reshape(n_mb, cfg.batch_size)
            self.params, self.opt_state, stopped, auxs, processed = self._epoch(
                self.params, self.opt_state, data, jnp.asarray(perm), lr, cfg.clip_range, self.frozen)
            processed = np.asarray(processed)
            auxs = {k: np.asarray(v)[processed] for k, v in auxs.items()}
            for k in stats:
                stats[k].extend(float(x) for x in auxs[k])
            approx_kls = [float(x) for x in auxs["approx_kl"]]  # SB3 reports the last epoch's
            if len(auxs["loss"]):
                last_loss = float(auxs["loss"][-1])
            self.n_updates += 1
            if bool(stopped):
                break
        v, ret = rollout["values"], rollout["returns"]
        var_y = np.var(ret)
        ev = float("nan") if var_y == 0 else float(1 - np.var(ret - v) / var_y)
        return {
            "train/entropy_loss": float(np.mean(stats["ent_loss"])),
            "train/policy_gradient_loss": float(np.mean(stats["pg_loss"])),
            "train/value_loss": float(np.mean(stats["v_loss"])),
            "train/approx_kl": float(np.mean(approx_kls)) if approx_kls else float("nan"),
            "train/clip_fraction": float(np.mean(stats["clip_frac"])),
            "train/loss": float(last_loss),
            "train/explained_variance": ev,
            "train/n_updates": float(self.n_updates),
            "train/clip_range": float(cfg.clip_range),
            "train/learning_rate": lr,
        }
