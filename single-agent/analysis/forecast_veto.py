"""k-step forecast veto ("rollout k") for the single-agent policy, the multi-agent team's window shield.

The forecast veto checks every proposed command before it is issued:
  - forecast: issue the commands of this decision exactly as the env would, advance the commanded
    step (the latest command offset + one 45 s interval), then k more 45 s steps with no further
    commands; collect the aircraft pairs with a loss of separation from now on;
  - trigger: a command is vetoed if its forecast has a pair that is neither in the forecast without
    that command (``silent``) nor in the forecast with no command at all (``baseline``);
  - replacement: the mask-valid clearance for the same flight that adds no such pair, fewest forecast
    pairs first, then the policy's own preference; else DO_NOTHING (always admissible).

The single agent issues one clearance per pick (at most two per interval with reselection), so the
"joint command" of a pick is the clearances already queued at this interval plus the new one, and the
acting agent is the picked flight. A loss of separation is the env's own (<= 3 NM and <= 500 ft,
``is_realised_los``), read from the simulator's LOSS_OF_SEPARATION events like the multi-agent ledger.

    veto = ForecastVeto(model, k=2); action, override = veto(env, obs, action)
"""
from __future__ import annotations

import numpy as np


class ForecastVeto:
    def __init__(self, model, k: int):
        self.model = model
        self.k = int(k)
        self.checked = self.overrides = self.forecasts = 0

    # -- forecast ----------------------------------------------------------------------------------
    def _pairs(self, env, cmds) -> frozenset:
        """Aircraft pairs with a loss of separation in the forecast of ``cmds`` (issued now)."""
        from flight_simulator import Infringement

        from utils.infringement_utils import is_realised_los, pair_severity

        self.forecasts += 1
        sim = env.simulator
        now = float(sim.world.age_in_seconds)
        stepped = sim.simulate_candidate(cmds)  # the commanded step, exactly as env.step advances it
        world = sim.advance_multiple_steps(steps=self.k, world=stepped) if self.k > 0 else stepped
        out = set()
        for infr in world.infringements:
            if isinstance(infr, Infringement.LOSS_OF_SEPARATION) and float(infr.timestamp_s) >= now:
                if is_realised_los(pair_severity(float(infr.horizontal_separation_nm), float(infr.vertical_separation_ft))):
                    out.add(tuple(sorted(str(x) for x in infr.aircraft_pair)))
        return frozenset(out)

    def _clearance_cmds(self, env, clr: int, callsign: str):
        from actions.actions import interpret_action_readonly

        return interpret_action_readonly(int(clr), callsign, env.simulator, env._route_trombone_store)

    def _preference(self, obs, a_air: int) -> np.ndarray:
        """The policy's masked clearance logits for flight slot a_air (no sampling, no RNG)."""
        import torch

        policy = self.model.policy
        with torch.no_grad():
            t, _ = policy.obs_to_tensor(obs)
            _air, clr, _again, _v = policy._masked_heads(t)
        return clr[0, a_air].cpu().numpy()

    # -- veto ----------------------------------------------------------------------------------------
    def __call__(self, env, obs, action):
        """(executed action, override dict or None)."""
        from actions.action_set import Action

        do_nothing = int(Action.DO_NOTHING)
        flat = np.asarray(action).ravel().astype(np.int64).copy()
        a_air, a_clr = int(flat[0]), int(flat[1])
        callsign = env._prev_aircraft_order[a_air]
        if a_clr == do_nothing or not callsign or callsign in getattr(env, "_picked", []):
            return flat, None
        self.checked += 1
        queued = list(getattr(env, "_pending_cmds", []) or [])
        with_q = env._with_queued
        current = self._pairs(env, with_q(self._clearance_cmds(env, a_clr, callsign)))
        silent = self._pairs(env, with_q(self._clearance_cmds(env, do_nothing, callsign)))
        baseline = silent if not queued else self._pairs(env, self._clearance_cmds(env, do_nothing, callsign))
        tolerated = silent | baseline
        if current <= tolerated:
            return flat, None
        mask = np.asarray(obs["mask_action_per_ac"][a_air]).astype(bool)
        valid = [int(c) for c in np.flatnonzero(mask)]
        pref = self._preference(obs, a_air)
        order = sorted(valid, key=lambda c: (-float(pref[c]), c))
        cands = []
        for rank, c in enumerate(order):
            if c == a_clr:
                continue
            pairs = silent if c == do_nothing else self._pairs(env, with_q(self._clearance_cmds(env, c, callsign)))
            if pairs <= tolerated:
                cands.append((len(pairs), rank, c))
        chosen = min(cands)[2] if cands else do_nothing
        self.overrides += 1
        flat[1] = chosen
        if chosen == do_nothing and flat.size > 2:
            flat[2] = 0  # a do-nothing closes the interval
        return flat, {"callsign": callsign, "proposed": a_clr, "chosen": chosen,
                      "new_pairs": sorted(current - tolerated)}
