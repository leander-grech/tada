"""Non-learning policies evaluated in the same MDP (evaluation_records.py --baseline).

Each is a drop-in for an SB3 model in score_windowed's loop: ``predict(obs, deterministic)`` returns the
windowed env's action ([aircraft slot, clearance] or, with reselection, [slot, clearance, again]).
The env is score_windowed's worker env (sw._ENV), read-only.

random  Uniformly random legal action: a selectable aircraft (the policy's own selection rule:
        mask_select, i.e. under-control slots not yet cleared this step), then a uniformly random
        clearance its mask allows (DO_NOTHING included), never a reselection pick. Sampled with the
        torch RNG that score_windowed seeds per (seed, attempt), so attempts differ and are repeatable.

greedy  A rule-based controller (one clearance per 45 s step, no learning, no search). It reads the
        env's do-nothing forecast (the same prediction the agent observes): the under-control flight
        with the largest predicted |landing deviation| beyond TOL_S gets one clearance that moves it
        the right way, by a fixed priority order of the procedure's levers:
          early (needs delay):    LENGTHEN_TROMBONE > SLOW_DOWN_LARGE > _MEDIUM > _SMALL > TURN_LEFT/RIGHT
          late  (needs advance):  DIRECT_TO_MERGE_POINT > VECTOR_TO_ILS > SHORTEN_TROMBONE > SPEED_UP_MEDIUM
                                  > SPEED_UP_SMALL > SKIP_4 > SKIP_3 > SKIP_2 > SKIP_1
        If that flight has no legal lever, the next flight is tried; otherwise DO_NOTHING. It ignores
        separation (the masks still apply) -- the naive "fix the worst flight" rule an AMAN-driven
        controller would follow without conflict reasoning.
"""
from __future__ import annotations

import numpy as np

TOL_S = 30.0  # inside +/-30 s a flight is left alone (half the +/-60 s on-time band)

DELAY_ORDER = ("LENGTHEN_TROMBONE", "SLOW_DOWN_LARGE", "SLOW_DOWN_MEDIUM", "SLOW_DOWN_SMALL",
               "TURN_LEFT_AND_RETURN_TO_ROUTE", "TURN_RIGHT_AND_RETURN_TO_ROUTE")
ADVANCE_ORDER = ("DIRECT_TO_MERGE_POINT", "VECTOR_TO_ILS", "SHORTEN_TROMBONE", "SPEED_UP_MEDIUM", "SPEED_UP_SMALL",
                 "SKIP_4_WAYPOINTS_NOT_NEXT", "SKIP_3_WAYPOINTS_NOT_NEXT", "SKIP_2_WAYPOINTS_NOT_NEXT", "SKIP_1_WAYPOINT")


def _selectable(obs) -> np.ndarray:
    from network.autoregressive_policy import _UNDER_CONTROL_COL

    if "mask_select" in obs:
        return np.asarray(obs["mask_select"], dtype=bool)
    vis = np.asarray(obs["mask_aircraft"], dtype=bool)
    uc = (np.asarray(obs["aircraft"])[:, _UNDER_CONTROL_COL] > 0.5) & vis
    return uc if uc.any() else vis


def _action(obs, slot: int, clr: int) -> np.ndarray:
    return np.array([slot, clr, 0] if "mask_select" in obs else [slot, clr], dtype=np.int64)


class RandomPolicy:
    name = "random"

    def predict(self, obs, deterministic: bool = False):
        import torch

        sel = np.flatnonzero(_selectable(obs))
        slot = int(sel[int(torch.randint(len(sel), (1,)))]) if len(sel) else 0
        legal = np.flatnonzero(np.asarray(obs["mask_action_per_ac"][slot], dtype=bool))
        clr = int(legal[int(torch.randint(len(legal), (1,)))]) if len(legal) else 0
        return _action(obs, slot, clr), None


class GreedyPolicy:
    name = "greedy"

    def __init__(self, env_getter):
        from actions.action_set import Action

        self._env = env_getter
        self._A = Action
        self._noop = int(Action.DO_NOTHING)
        self._delay = [int(getattr(Action, n)) for n in DELAY_ORDER if hasattr(Action, n)]
        self._advance = [int(getattr(Action, n)) for n in ADVANCE_ORDER if hasattr(Action, n)]

    def predict(self, obs, deterministic: bool = True):
        env = self._env()
        order = list(env._prev_aircraft_order)  # slot -> callsign ('' for an empty slot)
        land_t = env._predicted_landing_times()
        sel = _selectable(obs)
        cands = []
        for slot, cs in enumerate(order):
            if not cs or slot >= len(sel) or not sel[slot] or cs not in land_t or cs not in env._eta:
                continue
            dev = float(land_t[cs]) - float(env._eta[cs])  # negative = early, positive = late
            if abs(dev) > TOL_S:
                cands.append((abs(dev), slot, dev))
        for _absdev, slot, dev in sorted(cands, reverse=True):
            mask = np.asarray(obs["mask_action_per_ac"][slot], dtype=bool)
            for clr in (self._delay if dev < 0 else self._advance):
                if clr < len(mask) and mask[clr]:
                    return _action(obs, slot, clr), None
        slot = int(np.flatnonzero(sel)[0]) if sel.any() else 0
        return _action(obs, slot, self._noop), None


def make(name: str):
    from analysis import score_windowed as sw

    if name == "random":
        return RandomPolicy()
    if name == "greedy":
        return GreedyPolicy(lambda: sw._ENV)
    raise ValueError(f"unknown baseline {name!r}")
