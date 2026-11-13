"""The ATC autoregressive policy as pure JAX functions, over PyTorch's own parameters.

`params` is a flat dict keyed exactly like `ATCAutoregressivePolicy.state_dict()` (e.g.
"ar_net.encoder.ac_mlp.0.weight"), with every array in PyTorch's layout (Linear [out, in],
Conv1d [out, in, k], GRU gates stacked r|z|n). Converting between the two frameworks is
therefore a plain array copy -- nothing can be mis-mapped -- and a JAX-trained policy exports to
an SB3 checkpoint that every existing script loads unchanged (jax_ppo/export.py).

Each function mirrors network/rlm.py and network/autoregressive_policy.py line for line;
tests/test_jax_equivalence.py checks the outputs match to float32 precision.
"""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

Params = Dict[str, jnp.ndarray]
_NEG_INF = -1e8  # as network/autoregressive_policy.py: finite, so masked entries have no NaN grads
_LN_EPS = 1e-5  # torch.nn.LayerNorm default


def _linear(p: Params, name: str, x):
    y = x @ p[f"{name}.weight"].T
    b = p.get(f"{name}.bias")
    return y if b is None else y + b


def _layer_norm(p: Params, name: str, x):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + _LN_EPS) * p[f"{name}.weight"] + p[f"{name}.bias"]


def _conv1d(p: Params, name: str, x):
    """x: [N, C_in, W] -> [N, C_out, W]; kernel k (odd), padding k // 2 (torch Conv1d, cross-correlation).

    As one matrix product over the k shifted copies of the input (im2col), not a cuDNN convolution:
    these convolutions are tiny (k = 3 over 20 waypoints), cuDNN's default algorithms ran 5x slower
    than the autotuned ones on a laptop T550, and autotuning's trial workspaces (up to 2.7 GB) do not
    fit beside other processes on a 4 GB GPU. Same result to float32 rounding.
    """
    w = p[f"{name}.weight"]                                        # [C_out, C_in, k]
    k = w.shape[-1]
    xp = jnp.pad(x, ((0, 0), (0, 0), (k // 2, k // 2)))
    cols = jnp.stack([xp[:, :, j:j + x.shape[-1]] for j in range(k)], -1)   # [N, C_in, W, k]
    y = jnp.einsum("ncwk,ock->now", cols, w)
    return y + p[f"{name}.bias"][None, :, None]


def _gru_last_hidden(p: Params, name: str, x):
    """torch.nn.GRU (1 layer, batch_first, h0 = 0); returns the final hidden state [N, H]."""
    w_ih, w_hh = p[f"{name}.weight_ih_l0"], p[f"{name}.weight_hh_l0"]
    b_ih, b_hh = p[f"{name}.bias_ih_l0"], p[f"{name}.bias_hh_l0"]
    hidden = w_hh.shape[1]
    gi_all = x @ w_ih.T + b_ih  # [N, T, 3H], input projections for every step at once

    def step(h, gi):
        gh = h @ w_hh.T + b_hh
        i_r, i_z, i_n = jnp.split(gi, 3, axis=-1)
        h_r, h_z, h_n = jnp.split(gh, 3, axis=-1)
        r = jax.nn.sigmoid(i_r + h_r)
        z = jax.nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h, None

    h0 = jnp.zeros((x.shape[0], hidden), x.dtype)
    h, _ = jax.lax.scan(step, h0, jnp.swapaxes(gi_all, 0, 1))
    return h


def _attention(p: Params, name: str, x, mask_aircraft, heads: int):
    b, a, _ = x.shape
    q, k, v = (_linear(p, f"{name}.{n}", x) for n in ("q", "k", "v"))
    d = q.shape[-1] // heads
    shape = lambda t: jnp.swapaxes(t.reshape(b, a, heads, d), 1, 2)  # [B, H, A, d]
    q, k, v = shape(q), shape(k), shape(v)
    scores = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(jnp.asarray(d, x.dtype))
    keep = (mask_aircraft > 0.5)[:, None, None, :]
    scores = jnp.where(keep, scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1)
    attn = jnp.nan_to_num(attn, nan=0.0)  # a row with every key masked -> 0, as in torch
    pooled = jnp.swapaxes(attn @ v, 1, 2).reshape(b, a, heads * d)
    return _layer_norm(p, f"{name}.norm", x + _linear(p, f"{name}.out", pooled))


def encode(p: Params, obs: Dict[str, jnp.ndarray], heads: int):
    """ATCEncoder.encode: (aircraft_emb [B, A, H], mask_aircraft [B, A], context [B, 2H])."""
    e = "ar_net.encoder"
    aircraft = obs["aircraft"].astype(jnp.float32)
    glob = obs["global"].astype(jnp.float32)
    fp = obs["flight_plans_rel"].astype(jnp.float32)
    fp_mask = obs["flight_plan_mask"].astype(jnp.float32)
    mask_aircraft = obs["mask_aircraft"].astype(jnp.float32)
    hist = obs["action_histories"].astype(jnp.float32)
    b, a, w, f = fp.shape

    ac = jax.nn.relu(_layer_norm(p, f"{e}.ac_mlp.1", _linear(p, f"{e}.ac_mlp.0", aircraft)))

    x = jnp.swapaxes(fp.reshape(b * a, w, f), 1, 2)  # [BA, F, W]
    x = jax.nn.relu(_conv1d(p, f"{e}.fp_encoder.conv.0", x))
    x = jax.nn.relu(_conv1d(p, f"{e}.fp_encoder.conv.2", x))
    m = fp_mask.reshape(b * a, w)[:, None, :]
    x = (x * m).sum(-1) / jnp.maximum(m.sum(-1), 1.0)
    fp_emb = jax.nn.relu(_linear(p, f"{e}.fp_encoder.out", x)).reshape(b, a, -1)

    _, _, t, hd = hist.shape
    h_emb = _gru_last_hidden(p, f"{e}.action_history_gru", hist.reshape(b * a, t, hd)).reshape(b, a, -1)

    fused = jnp.concatenate([ac, fp_emb, h_emb], axis=-1)
    emb = jax.nn.relu(_layer_norm(p, f"{e}.aircraft_fuse_norm", _linear(p, f"{e}.aircraft_fuse", fused)))
    if f"{e}.aircraft_attention.q.weight" in p:
        emb = _attention(p, f"{e}.aircraft_attention", emb, mask_aircraft, heads)

    mexp = mask_aircraft[..., None]
    pooled = (emb * mexp).sum(1) / jnp.maximum(mexp.sum(1), 1.0)
    g = jax.nn.relu(_linear(p, f"{e}.glob_mlp.0", glob))
    return emb, mask_aircraft, jnp.concatenate([pooled, g], axis=-1)


def _mlp2(p: Params, name: str, x):
    return _linear(p, f"{name}.2", jax.nn.relu(_linear(p, f"{name}.0", x)))


def forward(p: Params, obs: Dict[str, jnp.ndarray], heads: int, under_control_col: int,
            do_nothing: int = 0):
    """Masked logits and values, as ATCAutoregressivePolicy._masked_heads.

    Returns (aircraft_logits [B, A], clearance_logits [B, A, C], again_logits [B, A, C] or None,
    values [B]). again_logits exist iff the parameters have the reselection head.
    """
    emb, mask_aircraft, ctx = encode(p, obs, heads)
    ac_logits, clr_logits, again_logits = policy_heads(p, obs, emb, mask_aircraft, ctx, under_control_col, do_nothing)
    values = _mlp2(p, "ar_net.value_head", ctx)[..., 0]
    return ac_logits, clr_logits, again_logits, values


def per_aircraft(emb, ctx):
    """[aircraft embedding | global context] per slot, the input of every per-aircraft head."""
    b, a, _ = emb.shape
    return jnp.concatenate([emb, jnp.broadcast_to(ctx[:, None, :], (b, a, ctx.shape[-1]))], axis=-1)


def policy_heads(p: Params, obs: Dict[str, jnp.ndarray], emb, mask_aircraft, ctx, under_control_col: int,
                 do_nothing: int = 0):
    """The masked policy logits of forward() from encoder features (encode())."""
    per_ac = per_aircraft(emb, ctx)
    ac_logits = _mlp2(p, "ar_net.aircraft_head", per_ac)[..., 0]
    clr_logits = _mlp2(p, "ar_net.clearance_head", per_ac)

    # Selectable aircraft: the env's mask_select under reselection; otherwise under-control
    # slots when any exist, else every visible slot.
    if "mask_select" in obs:
        selectable = obs["mask_select"].astype(jnp.float32)
    else:
        under = (obs["aircraft"][..., under_control_col] > 0.5).astype(jnp.float32) * mask_aircraft
        any_under = under.sum(-1, keepdims=True) > 0.5
        selectable = jnp.where(any_under, under, mask_aircraft)
    ac_logits = jnp.where(selectable < 0.5, _NEG_INF, ac_logits)
    clr_logits = jnp.where(obs["mask_action_per_ac"] < 0.5, _NEG_INF, clr_logits)

    again_logits = None
    if "ar_net.again_head.0.weight" in p:
        again_logits = _mlp2(p, "ar_net.again_head", per_ac)
        allowed = (obs["mask_again"][:, 0] > 0.5)[:, None, None] & (jnp.arange(again_logits.shape[-1]) != do_nothing)
        again_logits = jnp.where(allowed, again_logits, _NEG_INF)
    return ac_logits, clr_logits, again_logits


def _bernoulli_logp(logits, x):
    """log p(x) for x in {0, 1} under Bernoulli(sigmoid(logits)), as torch's Bernoulli."""
    return -(jnp.maximum(logits, 0.0) - logits * x + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _bernoulli_entropy(logits):
    p1 = jax.nn.sigmoid(logits)
    return jnp.maximum(logits, 0.0) - logits * p1 + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def log_prob_entropy(ac_logits, clr_logits, actions, again_logits=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """log pi(a) of joint actions and the exact autoregressive entropy
    H(aircraft) + sum_a p(a) [H(clearance | a) + sum_c p(c | a) H(again | a, c)],
    as ATCAutoregressivePolicy.evaluate_actions."""
    ac_logp = jax.nn.log_softmax(ac_logits, -1)
    clr_logp_all = jax.nn.log_softmax(clr_logits, -1)  # [B, A, C]
    a_air, a_clr = actions[:, 0], actions[:, 1]
    idx = jnp.arange(ac_logits.shape[0])
    logp = ac_logp[idx, a_air] + clr_logp_all[idx, a_air, a_clr]
    p_ac = jnp.exp(ac_logp)
    h_ac = -(p_ac * ac_logp).sum(-1)
    p_clr = jnp.exp(clr_logp_all)
    h_clr = -(p_clr * clr_logp_all).sum(-1)  # [B, A]
    if again_logits is not None:
        logp = logp + _bernoulli_logp(again_logits[idx, a_air, a_clr], actions[:, 2].astype(jnp.float32))
        h_clr = h_clr + (p_clr * _bernoulli_entropy(again_logits)).sum(-1)
    return logp, h_ac + (p_ac * h_clr).sum(-1)


def sample(key, ac_logits, clr_logits, again_logits=None, deterministic: bool = False):
    """Aircraft first, then a clearance conditioned on it, then (with reselection) the again
    flag conditioned on both. Returns (actions [B, 2 or 3], log_prob)."""
    k1, k2 = jax.random.split(key)
    if deterministic:
        a_air = jnp.argmax(ac_logits, -1)
    else:
        a_air = jax.random.categorical(k1, ac_logits, axis=-1)
    idx = jnp.arange(ac_logits.shape[0])
    chosen = clr_logits[idx, a_air]
    a_clr = jnp.argmax(chosen, -1) if deterministic else jax.random.categorical(k2, chosen, axis=-1)
    logp = jax.nn.log_softmax(ac_logits, -1)[idx, a_air] + jax.nn.log_softmax(chosen, -1)[idx, a_clr]
    if again_logits is None:
        return jnp.stack([a_air, a_clr], -1), logp
    l_again = again_logits[idx, a_air, a_clr]
    if deterministic:
        a_again = (l_again > 0).astype(jnp.int32)
    else:
        a_again = jax.random.bernoulli(jax.random.fold_in(key, 3), jax.nn.sigmoid(l_again)).astype(jnp.int32)
    logp = logp + _bernoulli_logp(l_again, a_again.astype(jnp.float32))
    return jnp.stack([a_air, a_clr, a_again.astype(a_air.dtype)], -1), logp
