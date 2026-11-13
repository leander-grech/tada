"""Critic-guided one-step lookahead: an inference-time policy improvement step.

At every decision, take the policy's K most likely (aircraft, clearance) actions -- always
including the deterministic policy's own choice -- try each for one real env step on a snapshot
of the environment, score it

    Q(a) = clip(r(s, a) / sigma_R) + gamma * V(s'_a)        (V = 0 after a true termination)

with the policy's OWN value function V, and act with the best. sigma_R is the return scale PPO's
VecNormalize learned, read from the checkpoint's vecnormalize file, so r and V are in the units the
critic was trained in -- which also needs the env to pay the reward the run trained on
(reward_mode_for). Under reselection each candidate carries the policy's own again flag for it,
and a zero-time pick (info["zero_time"]) is discounted by 1, as in training. This is one step of policy iteration on the learned critic: it changes the
policy's choice only where its own value estimate says another likely action is better.

Deployable: it needs only the simulator, which the environment already runs, and K env steps
per 45 s decision (~30 ms each). It is not best-of-N over whole episodes, which would need the
future to be replayable.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch



def vecnormalize_path_for(model_path: str) -> Path:
    """The VecNormalize statistics saved with a checkpoint (final_model or ppo_tada_N_steps)."""
    p = Path(model_path)
    m = re.match(r"ppo_tada_(\d+)_steps\.zip$", p.name)
    if m:
        return p.with_name(f"ppo_tada_vecnormalize_{m.group(1)}_steps.pkl")
    if p.name == "final_model.zip":
        return p.with_name("vecnormalize.pkl")
    if p.name == "best_model.zip":
        return p.parent.parent / "vecnormalize.pkl"  # closest available; best has none of its own
    raise FileNotFoundError(f"no vecnormalize file known for {model_path}")


def reward_mode_for(model_path: str) -> str:
    """The windowed reward mode a checkpoint's run trained on (its run_meta.json), else the
    windowed default."""
    from config.windowed_config import WindowedConfig

    p = Path(model_path).resolve()
    for d in (p.parent, p.parent.parent):
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                mode = (json.loads(meta.read_text()).get("windowed_config") or {}).get("reward_mode")
            except Exception:  # noqa: BLE001
                mode = None
            if mode:
                return str(mode)
    return str(WindowedConfig.REWARD_MODE)


# Env keys that change what the agent sees or can do (not only its reward): evaluation has to build
# the env with the run's values, or the policy acts in a different MDP from the one it learned.
ENV_KEYS = ("release_locked", "release_dev_s", "use_case")
REWARD_KEYS = ("pbrs_violation_penalty", "pbrs_flat_potentials", "landing_brackets", "pay_at_release",
               "pbrs_flat_deviation")


def _run_windowed_config(model_path: str) -> dict:
    p = Path(model_path).resolve()
    for d in (p.parent, p.parent.parent):
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                return json.loads(meta.read_text()).get("windowed_config") or {}
            except Exception:  # noqa: BLE001
                return {}
    return {}


def env_config_for(model_path: str) -> dict:
    """The env keys that set the MDP a checkpoint's run trained in (ENV_KEYS), from run_meta.json."""
    wcfg = _run_windowed_config(model_path)
    return {k: wcfg[k] for k in ENV_KEYS if k in wcfg}


def reward_config_for(model_path: str) -> dict:
    """The env keys that set the reward a checkpoint's run trained on (run_meta.json):
    reward_mode, the run's ENV_KEYS, and the REWARD_KEYS the run changed."""
    wcfg = _run_windowed_config(model_path)
    out = {"reward_mode": reward_mode_for(model_path), **env_config_for(model_path)}
    for k in REWARD_KEYS:
        if k not in wcfg or wcfg[k] in (False, None):
            continue
        v = wcfg[k]
        # hashable, so score_windowed can compare the models' configs in a set
        out[k] = tuple(tuple(float(x) for x in b) for b in v) if k == "landing_brackets" else v
    return out


class LookaheadAgent:
    def __init__(self, model, model_path: str, k: int, gamma: float):
        self.model = model
        self.k = int(k)
        self.gamma = float(gamma)
        with open(vecnormalize_path_for(model_path), "rb") as fh:
            vn = pickle.load(fh)
        self.sigma = float(np.sqrt(vn.ret_rms.var + vn.epsilon))
        self.clip = float(vn.clip_reward)
        self.overrides = 0
        self.decisions = 0

    def _candidates(self, obs) -> list[tuple[int, ...]]:
        policy = self.model.policy
        obs_t, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            a_logits, c_logits, again_logits, _ = policy._masked_heads(obs_t)
            det, _ = self.model.predict(obs, deterministic=True)
        det = tuple(int(x) for x in np.asarray(det).ravel())
        joint = (torch.log_softmax(a_logits[0], -1)[:, None] + torch.log_softmax(c_logits[0], -1)).flatten()
        n_clr = c_logits.shape[-1]
        cands = [det]
        for idx in torch.argsort(joint, descending=True).tolist():
            if len(cands) >= self.k:
                break
            if float(joint[idx]) < -1e6:  # masked
                break
            pair = (idx // n_clr, idx % n_clr)
            if again_logits is not None:  # the deterministic policy's own flag for this pair
                pair = pair + (int(float(again_logits[0, pair[0], pair[1]]) > 0),)
            if pair not in cands:
                cands.append(pair)
        return cands

    def _value(self, obs) -> float:
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            return float(self.model.policy.predict_values(obs_t).ravel()[0])

    def act(self, env, obs) -> np.ndarray:
        cands = self._candidates(obs)
        self.decisions += 1
        if len(cands) == 1:
            return np.array(cands[0], dtype=np.int64)
        snap = env.snapshot()
        best, best_q = cands[0], -np.inf
        for a in cands:
            env.restore(snap)
            try:
                obs2, r, te, tr, info = env.step(np.array(a, dtype=np.int64))
            except Exception:  # a mask-valid clearance the interpreter cannot execute here
                continue
            q = float(np.clip(r / self.sigma, -self.clip, self.clip))
            if not te:
                q += (1.0 if info.get("zero_time", False) else self.gamma) * self._value(obs2)
            if q > best_q:
                best, best_q = a, q
        env.restore(snap)
        self.overrides += int(best != cands[0])
        return np.array(best, dtype=np.int64)
