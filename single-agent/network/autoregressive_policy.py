"""Autoregressive actor-critic policy for the MultiDiscrete([N_aircraft, N_clearances]) path.

Two factored action heads:

1. an aircraft head -> Categorical over aircraft (masked by ``mask_aircraft``, restricted to
   aircraft under control when any are -- see _masked_logits), then
2. a clearance head conditioned on the *sampled* aircraft -> Categorical over clearances
   (masked by ``mask_action_per_ac[chosen_aircraft]``).

With reselection (Config.MAX_PICKS > 1, action MultiDiscrete([A, C, 2])) a third stage follows:

3. an "again" flag conditioned on the sampled aircraft AND clearance -> Bernoulli: 1 asks the
   env for another pick before the clock advances. Allowed only where ``mask_again`` is set
   (budget left and another aircraft to pick) and the clearance is a real one: DO_NOTHING
   always ends the step. Where it is not allowed the flag is 0 with probability 1, and adds
   nothing to the log-probability or the entropy. The aircraft head then reads the env's
   ``mask_select``, which also excludes aircraft already cleared at this step.

Both masks are read directly from the observation (the env always exposes them and
``DO_NOTHING`` is always valid, so no masked categorical is ever all-``-inf``). This is why
the policy uses SB3 ``PPO`` with clearance masks conditioned on the sampled aircraft.
"""

from typing import Dict, Optional, Tuple, Union

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Bernoulli, Categorical

from actions.actions import Action
from models.observations import AircraftScalarFeatures
from network.rlm import ATCAutoregressiveNetwork

# Column of the under_control flag in obs["aircraft"], derived so it cannot drift.
_UNDER_CONTROL_COL = list(AircraftScalarFeatures.__dataclass_fields__).index("under_control")

# Large finite negative (not -inf) so masked logits contribute ~0 probability without
# producing NaN gradients; masked entries are constants, so no gradient flows through them.
_NEG_INF = -1e8

_DO_NOTHING = int(Action.DO_NOTHING)

TensorDict = Dict[str, torch.Tensor]


class ATCAutoregressivePolicy(ActorCriticPolicy):
    """Custom ActorCriticPolicy with autoregressive (aircraft -> clearance) heads.

    The default ``mlp_extractor``/``action_net``/``value_net`` pipeline is bypassed: this
    policy owns an :class:`ATCAutoregressiveNetwork` that consumes the raw Dict observation
    and produces aircraft logits ``[B, A]``, clearance logits ``[B, A, C]`` and a value
    ``[B]``. The default ``FlattenExtractor`` is instantiated by the base class but never used.
    """

    def __init__(self, observation_space, action_space, lr_schedule, head_hidden: int = 64, **kwargs):
        # Stored before super().__init__ because it calls self._build().
        self._head_hidden = int(head_hidden)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        # Replace the default actor/critic pipeline with the autoregressive network. The "again"
        # head exists iff the action space has the third component, so a checkpoint's own saved
        # action space decides it and every (aircraft, clearance) checkpoint loads unchanged.
        self.ar_net = ATCAutoregressiveNetwork(head_hidden=self._head_hidden, again=self.has_again)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    @property
    def has_again(self) -> bool:
        nvec = getattr(self.action_space, "nvec", None)
        return nvec is not None and len(nvec) == 3

    # ------------------------------------------------------------------
    # Masked logits
    # ------------------------------------------------------------------

    def _masked_logits(
        self, obs: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the network and mask the logits using the masks carried in the obs.

        Returns (aircraft_logits [B, A], clearance_logits [B, A, C], values [B]).
        """
        aircraft_logits, clearance_logits, _again_logits, values = self._masked_heads(obs)
        return aircraft_logits, clearance_logits, values

    def _masked_heads(
        self, obs: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """As _masked_logits, plus the masked again logits [B, A, C] (None without the head)."""
        aircraft_logits, clearance_logits, again_logits, values = self.ar_net.forward_all(obs)

        mask_aircraft = obs["mask_aircraft"].float()        # [B, A]
        mask_clearance = obs["mask_action_per_ac"].float()  # [B, A, C]

        # Which slots the aircraft head may PICK. A visible aircraft that is not yet under
        # control (the windowed env keeps not-yet-spawned flights in their queue slot, so the
        # agent can plan around them) can only be left alone, so picking it would silently spend
        # the step on a no-op. Restrict selection to under-control slots, falling back to every
        # visible slot on rows where none is under control (so no row is ever all-masked).
        #
        # With reselection the env states the selectable slots itself (``mask_select``): the
        # same rule, minus the aircraft already cleared at this step.
        if "mask_select" in obs:
            selectable = obs["mask_select"].float()
        else:
            under_control = (obs["aircraft"][..., _UNDER_CONTROL_COL] > 0.5).float() * mask_aircraft
            any_under_control = under_control.sum(dim=-1, keepdim=True) > 0.5
            selectable = torch.where(any_under_control, under_control, mask_aircraft)

        aircraft_logits = aircraft_logits.masked_fill(selectable < 0.5, _NEG_INF)
        clearance_logits = clearance_logits.masked_fill(mask_clearance < 0.5, _NEG_INF)

        if again_logits is not None:
            # again=1 is possible only with budget left (mask_again) and a real clearance.
            allowed = (obs["mask_again"].float()[:, 0] > 0.5)[:, None, None].expand_as(again_logits).clone()
            allowed[..., _DO_NOTHING] = False
            again_logits = again_logits.masked_fill(~allowed, _NEG_INF)

        return aircraft_logits, clearance_logits, again_logits, values

    @staticmethod
    def _gather_clearance(clearance_logits: torch.Tensor, aircraft: torch.Tensor) -> torch.Tensor:
        """Select the clearance logits row [B, C] for the chosen aircraft [B]."""
        batch_idx = torch.arange(clearance_logits.shape[0], device=clearance_logits.device)
        return clearance_logits[batch_idx, aircraft]

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def forward(
        self, obs: TensorDict, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aircraft_logits, clearance_logits, again_logits, values = self._masked_heads(obs)

        aircraft_dist = Categorical(logits=aircraft_logits)
        if deterministic:
            a_air = torch.argmax(aircraft_logits, dim=-1)
        else:
            a_air = aircraft_dist.sample()

        chosen_clearance_logits = self._gather_clearance(clearance_logits, a_air)
        clearance_dist = Categorical(logits=chosen_clearance_logits)
        if deterministic:
            a_clr = torch.argmax(chosen_clearance_logits, dim=-1)
        else:
            a_clr = clearance_dist.sample()

        log_prob = aircraft_dist.log_prob(a_air) + clearance_dist.log_prob(a_clr)
        if again_logits is None:
            actions = torch.stack([a_air, a_clr], dim=-1)  # [B, 2]
            return actions, values, log_prob

        batch_idx = torch.arange(again_logits.shape[0], device=again_logits.device)
        again_dist = Bernoulli(logits=again_logits[batch_idx, a_air, a_clr])
        if deterministic:
            a_again = (again_dist.logits > 0).float()
        else:
            a_again = again_dist.sample()
        log_prob = log_prob + again_dist.log_prob(a_again)
        actions = torch.stack([a_air, a_clr, a_again.long()], dim=-1)  # [B, 3]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: TensorDict, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aircraft_logits, clearance_logits, again_logits, values = self._masked_heads(obs)

        actions = actions.long()
        a_air = actions[:, 0]
        a_clr = actions[:, 1]

        aircraft_dist = Categorical(logits=aircraft_logits)
        chosen_clearance_logits = self._gather_clearance(clearance_logits, a_air)
        clearance_dist = Categorical(logits=chosen_clearance_logits)

        log_prob = aircraft_dist.log_prob(a_air) + clearance_dist.log_prob(a_clr)

        # Exact expected entropy over the autoregressive tree:
        #   H(aircraft) + sum_a p(a) * H(clearance | a)
        # Cheap because A is small (10) and a function of obs only, so it is consistent
        # between forward() and evaluate_actions().
        aircraft_probs = aircraft_dist.probs                       # [B, A]
        clearance_entropy_all = Categorical(logits=clearance_logits).entropy()  # [B, A]
        if again_logits is not None:
            # + sum_a p(a) sum_c p(c | a) H(again | a, c); a disallowed flag has entropy 0.
            batch_idx = torch.arange(again_logits.shape[0], device=again_logits.device)
            again_dist = Bernoulli(logits=again_logits[batch_idx, a_air, a_clr])
            log_prob = log_prob + again_dist.log_prob(actions[:, 2].float())
            again_entropy_all = Bernoulli(logits=again_logits).entropy()          # [B, A, C]
            clearance_probs_all = torch.softmax(clearance_logits, dim=-1)         # [B, A, C]
            clearance_entropy_all = clearance_entropy_all + (clearance_probs_all * again_entropy_all).sum(-1)
        entropy = aircraft_dist.entropy() + (aircraft_probs * clearance_entropy_all).sum(dim=1)

        return values, log_prob, entropy

    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        _aircraft_logits, _clearance_logits, values = self.ar_net(obs)
        return values

    def _predict(
        self, observation: Union[torch.Tensor, TensorDict], deterministic: bool = False
    ) -> torch.Tensor:
        actions, _values, _log_prob = self.forward(observation, deterministic=deterministic)
        return actions


def warm_start_policy(dst_policy, src_policy) -> list:
    """Copy ``src_policy``'s weights into ``dst_policy`` (``main_jax.py --init-weights``).

    Exact copy, except two supported extensions:
      * the first per-aircraft layer widening (12 -> 12 + SEQUENCE_FEATURES with
        TADA_SEQUENCE_OBS=1, and + PICK_FEATURES with TADA_MAX_PICKS > 1). Its old columns are
        copied and the new ones zeroed, so the new columns start with no influence;
      * a fresh "again" head (reselection) when the source has none. It keeps its own init,
        whose flag probability is the same small value everywhere.
    With the new columns at zero on a step's first pick and again=0, the initial policy computes
    exactly the source's function. Any other mismatch raises, rather than training from a half-loaded network.
    Returns a description of each widened or added tensor.
    """
    src_state = src_policy.state_dict()
    dst_state = dst_policy.state_dict()
    added = {k for k in set(dst_state) - set(src_state) if k.startswith("ar_net.again_head.")}
    if set(src_state) != set(dst_state) - added:
        raise RuntimeError(f"parameter names differ: {sorted(set(src_state) ^ (set(dst_state) - added))[:6]}")
    widened = [f"{k} {tuple(dst_state[k].shape)} (new, own init)" for k in sorted(added)]
    with torch.no_grad():
        for key, dst in dst_state.items():
            if key in added:
                continue
            src = src_state[key]
            if src.shape == dst.shape:
                dst.copy_(src)
            elif (key == "ar_net.encoder.ac_mlp.0.weight" and src.dim() == 2
                  and src.shape[0] == dst.shape[0] and src.shape[1] < dst.shape[1]):
                dst.zero_()
                dst[:, : src.shape[1]].copy_(src)
                widened.append(f"{key} {tuple(src.shape)} -> {tuple(dst.shape)}")
            else:
                raise RuntimeError(f"cannot warm-start {key}: {tuple(src.shape)} -> {tuple(dst.shape)}")
    dst_policy.load_state_dict(dst_state, strict=True)
    return widened
