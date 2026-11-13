"""Select the clearance set and checkpoint compatibility settings."""

from config.config import Config

_SET = str(getattr(Config, "ACTION_SET", "v2")).lower()

if _SET == "v2":
    from actions.actions_v2 import (  # noqa: F401
        ACTION_DEFINITIONS_BASE,
        STORE_MUTATING_CLEARANCES,
        Action,
        get_action_definitions,
        interpret_action,
    )
    from actions.action_mask_v2 import compute_action_mask  # noqa: F401
elif _SET == "v2pms":
    # Point merge (BGY, use case 2): v2's width, with DIRECT_TO_MERGE_POINT in slot 14 and the
    # flight flying its whole sequencing leg (actions_v2pms.py). Clearances are procedure
    # dependent, so a PMS run uses this set and a TMB (trombone) run uses v2.
    from actions.actions_v2pms import (  # noqa: F401
        ACTION_DEFINITIONS_BASE,
        STORE_MUTATING_CLEARANCES,
        Action,
        get_action_definitions,
        interpret_action,
    )
    from actions.action_mask_v2pms import compute_action_mask  # noqa: F401
else:
    raise ValueError(f"Config.ACTION_SET must be 'v2' or 'v2pms', got {Config.ACTION_SET!r}")

# Optional procedure-dependent blocking mask laid over the set (TADA_AIRPORT_MASK, read at import
# like ACTION_SET; recorded in run_meta.json as "airport_mask" and checked when a checkpoint is
# loaded). A mask specifies allowed (1) and blocked (0) clearances for each procedure.
# The set fixes the procedure: v2 = MXP trombone, v2pms = BGY point merge.
AIRPORT_MASKS = {
    "procedure": {
        "v2": [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],     # MXP: no SLOW_DOWN_SMALL, SPEED_UP_MEDIUM
        "v2pms": [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],  # BGY: no turns, LENGTHEN, SKIP_4
    },
}
AIRPORT_MASK = str(getattr(Config, "AIRPORT_MASK", "") or "").lower() or None
if AIRPORT_MASK is not None:
    if AIRPORT_MASK not in AIRPORT_MASKS or _SET not in AIRPORT_MASKS[AIRPORT_MASK]:
        raise ValueError(f"TADA_AIRPORT_MASK={AIRPORT_MASK!r} has no mask for action set {_SET!r}")
    import numpy as _np

    _BLOCK = _np.asarray(AIRPORT_MASKS[AIRPORT_MASK][_SET], dtype=bool)
    _compute_action_mask_unmasked = compute_action_mask

    def compute_action_mask(*args, **kwargs):  # noqa: F811
        mask = _compute_action_mask_unmasked(*args, **kwargs) & _BLOCK
        if not mask.any():  # only blocked clearances were legal and DO_NOTHING was gated off
            mask[int(Action.DO_NOTHING)] = True
        return mask

# Config states the size as a literal (it cannot import this module without a cycle), so the
# two can drift. Fail loudly at import rather than silently mis-size the policy head.
if len(Action) != int(Config.NUM_ACTIONS):
    raise AssertionError(
        f"action set {_SET!r} has {len(Action)} clearances but "
        f"Config.NUM_ACTIONS is {Config.NUM_ACTIONS}"
    )

ACTION_SET_VERSION = _SET


def interpret_action_readonly(clr: int, call_sign: str, simulator, route_trombone_store):
    """Interpret a clearance without mutating live episode state.

    Some clearances mutate the RouteSectionStore as a side effect of being interpreted --
    lengthening pulls levels out of it, and in v2 shortening pushes them back -- so any
    read-only "what would this clearance do?" probe has to hand ``interpret_action`` a copy.

    RouteSectionStore.clone() preserves the live state, including PyO3 route points.
    """
    mutating = {int(a) for a in STORE_MUTATING_CLEARANCES}
    store = route_trombone_store.clone() if int(clr) in mutating else route_trombone_store
    return interpret_action(clr, call_sign, simulator, store)


def checkpoint_clearance_count(model_path) -> int | None:
    """How many clearances a saved policy was trained with, or None if undeterminable.

    Read straight out of the zip without constructing a model, from the width of the
    clearance head's output layer.
    """
    import io
    import zipfile

    try:
        import torch

        with zipfile.ZipFile(str(model_path)) as zf:
            state = torch.load(
                io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True
            )
        for key in ("ar_net.clearance_head.2.bias", "clearance_head.2.bias"):
            if key in state:
                return int(state[key].shape[0])
    except Exception:  # noqa: BLE001 - a best-effort diagnostic must never be the failure
        return None
    return None


def sync_attention_to_checkpoint(model_path) -> None:
    """Set Config.USE_AIRCRAFT_ATTENTION to match a saved policy before it is loaded.

    The attention flag changes the network without changing observation or action spaces.
    Inspect the saved parameters and leave the flag unchanged if they cannot be read.
    """
    import io
    import zipfile

    try:
        import torch

        with zipfile.ZipFile(str(model_path)) as zf:
            state = torch.load(
                io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True
            )
    except Exception:  # noqa: BLE001
        return
    Config.USE_AIRCRAFT_ATTENTION = any("aircraft_attention" in key for key in state)


def checkpoint_obs_width(model_path) -> int | None:
    """Per-aircraft observation width a saved policy was trained with (12, +4 with the
    AMAN-sequence columns, +1 with reselection), from its first per-aircraft layer; None if
    undeterminable."""
    import io
    import zipfile

    try:
        import torch

        with zipfile.ZipFile(str(model_path)) as zf:
            state = torch.load(
                io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True
            )
        return int(state["ar_net.encoder.ac_mlp.0.weight"].shape[1])
    except Exception:  # noqa: BLE001
        return None


def load_policy_at_own_width(loader, model_path, **kwargs):
    """``loader.load(model_path)`` with the network built at the checkpoint's OWN observation
    width, whatever the active config says -- the source side of a widened warm start
    (``main_jax.py --init-weights`` with TADA_SEQUENCE_OBS=1). Config is restored afterwards."""
    width = checkpoint_obs_width(model_path)
    active = Config.AIRCRAFT_SCALAR_LEN
    try:
        if width is not None:
            Config.AIRCRAFT_SCALAR_LEN = width
        return loader.load(model_path, **kwargs)
    finally:
        Config.AIRCRAFT_SCALAR_LEN = active


def checkpoint_max_picks(model_path) -> int:
    """Picks per step a saved policy was trained with: 1 without the reselection ("again")
    head; with it, the run's recorded max_picks (run_meta.json of the experiment the checkpoint
    belongs to), else TADA_MAX_PICKS if set above 1, else 2."""
    import io
    import json
    import os
    import zipfile
    from pathlib import Path

    try:
        import torch

        with zipfile.ZipFile(str(model_path)) as zf:
            state = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        return int(Config.MAX_PICKS)
    if not any(k.startswith("ar_net.again_head.") for k in state):
        return 1
    path = Path(model_path).resolve()
    for d in (path.parent, path.parent.parent):
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                value = json.loads(meta.read_text()).get("max_picks")
            except Exception:  # noqa: BLE001
                value = None
            if value:
                return int(value)
    if int(os.getenv("TADA_MAX_PICKS", "1")) > 1:
        return int(os.environ["TADA_MAX_PICKS"])
    return 2


def run_action_set(model_path) -> str | None:
    """The clearance set a checkpoint's run trained with, from its run_meta.json ("action_set",
    if run_meta.json is found); otherwise None."""
    import json
    from pathlib import Path

    path = Path(model_path).resolve()
    for d in (path.parent, path.parent.parent):
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                return str(json.loads(meta.read_text()).get("action_set") or "v2").lower()
            except Exception:  # noqa: BLE001
                return None
    return None


def run_airport_mask(model_path) -> str | None:
    """The airport mask a checkpoint's run trained with (run_meta.json "airport_mask"), or None."""
    import json
    from pathlib import Path

    path = Path(model_path).resolve()
    for d in (path.parent, path.parent.parent):
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                mask = json.loads(meta.read_text()).get("airport_mask") or None
                return "procedure" if mask == "monad" else mask
            except Exception:  # noqa: BLE001
                return None
    return None


def check_run_action_set(model_path) -> None:
    """Refuse to load a checkpoint under a different clearance set of the same width.

    v2 (trombone) and v2pms (point merge) both have 15 clearances, so the width check in
    check_checkpoint_compatible cannot tell them apart. Without this, a PMS policy scored under
    v2 would have its DIRECT_TO_MERGE_POINT read as VECTOR_TO_ILS, on a route that was cut
    instead of flying the leg. Set TADA_ACTION_SET_OVERRIDE=1 to cross sets deliberately (a
    zero-shot transfer test).
    """
    import os

    trained = run_action_set(model_path)
    if os.getenv("TADA_ACTION_SET_OVERRIDE") == "1":
        return
    trained_mask = run_airport_mask(model_path)
    if trained is not None and trained_mask != AIRPORT_MASK:
        raise SystemExit(
            f"\nAirport-mask mismatch: {model_path}\n  trained with TADA_AIRPORT_MASK={trained_mask or ''}, but the "
            f"active mask is {AIRPORT_MASK or 'none'}.\n  Re-run with TADA_AIRPORT_MASK={trained_mask or ''} "
            f"(or TADA_ACTION_SET_OVERRIDE=1 for a deliberate cross-set test).\n")
    if trained is None or trained == ACTION_SET_VERSION:
        return
    raise SystemExit(
        f"\nClearance-set mismatch: {model_path}\n  trained with TADA_ACTION_SET={trained}, but the "
        f"active set is {ACTION_SET_VERSION!r}.\n  Re-run with TADA_ACTION_SET={trained} "
        f"(or TADA_ACTION_SET_OVERRIDE=1 for a deliberate cross-set test).\n")


def sync_mask_frame_to_checkpoint(model_path) -> None:
    """Match Config.MASK_FRAME_FIX and Config.TURN_MASK_FIX to the checkpoint's run (run_meta.json
    "mask_frame_fix" / "turn_mask_fix"; absent = a run from before the fix = False), here and in worker
    processes (TADA_MASK_FRAME_FIX / TADA_TURN_MASK_FIX). Left unchanged when no run_meta.json is found."""
    import json
    import os
    from pathlib import Path

    path = Path(model_path).resolve()
    for d in (path.parent, path.parent.parent):
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                fixed = bool(m.get("mask_frame_fix", False))
                turn_fixed = bool(m.get("turn_mask_fix", False))
            except Exception:  # noqa: BLE001
                return
            Config.MASK_FRAME_FIX = fixed
            os.environ["TADA_MASK_FRAME_FIX"] = "1" if fixed else "0"
            Config.TURN_MASK_FIX = turn_fixed
            os.environ["TADA_TURN_MASK_FIX"] = "1" if turn_fixed else "0"
            return


def sync_obs_width_to_checkpoint(model_path) -> None:
    """Match Config.AIRCRAFT_SCALAR_LEN / SEQUENCE_OBS / MAX_PICKS to a saved policy before
    building envs.

    A policy trained with the AMAN-sequence columns (TADA_SEQUENCE_OBS=1) has 4 extra inputs on
    its first per-aircraft layer, and one trained with reselection (TADA_MAX_PICKS > 1) has an
    "again" head and 1 extra input: 12, 13, 16 or 17 wide. Sets TADA_SEQUENCE_OBS and
    TADA_MAX_PICKS in the process environment too, so worker processes that re-import config
    agree. Read from the zip.
    """
    import os

    check_run_action_set(model_path)
    sync_mask_frame_to_checkpoint(model_path)
    width = checkpoint_obs_width(model_path)
    if width is None:
        return
    picks = checkpoint_max_picks(model_path)
    base = 12 + (int(Config.PICK_FEATURES) if picks > 1 else 0)
    seq = width == base + int(Config.SEQUENCE_FEATURES)
    Config.SEQUENCE_OBS = seq
    Config.MAX_PICKS = picks
    Config.AIRCRAFT_SCALAR_LEN = width
    os.environ["TADA_SEQUENCE_OBS"] = "1" if seq else "0"
    os.environ["TADA_MAX_PICKS"] = str(picks)


def check_checkpoint_compatible(model_path) -> None:
    """Fail early and legibly when a checkpoint's action set differs from the active one.

    Without this the mismatch surfaces as a raw torch state_dict error naming
    ``clearance_head.2.weight`` and ``action_history_gru.weight_ih_l0``, which says nothing
    about action sets and gives no hint that an environment variable is the remedy. Since
    ACTION_SET is read at import time, the fix cannot be a CLI flag, so the message has to
    spell out the invocation.
    """
    found = checkpoint_clearance_count(model_path)
    if found is None or found == int(Config.NUM_ACTIONS):
        return

    wanted = {size: name for name, size in Config._ACTION_SET_SIZES.items()}.get(found)
    remedy = (
        f"TADA_ACTION_SET={wanted} " if wanted else "TADA_ACTION_SET=<the set it was trained on> "
    )
    raise SystemExit(
        f"\nCheckpoint/action-set mismatch.\n"
        f"  {model_path}\n"
        f"  was trained with {found} clearances, but ACTION_SET={ACTION_SET_VERSION!r} "
        f"has {Config.NUM_ACTIONS}.\n\n"
        f"Re-run with:\n"
        f"  {remedy}python <your command>\n\n"
        f"(ACTION_SET is consumed at import time -- it sizes the action space, the clearance\n"
        f" head and the action-history one-hot -- so it cannot be a command-line flag.)\n"
    )
