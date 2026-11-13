"""Run naming, metadata and learning-rate schedules for the JAX trainer."""

import json
import math
import re
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.utils import get_latest_run_id
from config.config import Config

ROLLOUT_BUFFER_SIZE = 4096


GAMMA = 0.998


def _next_run_name(base: str = "atc_run_1") -> str:
    """Next run id, continuing the sequence across tb_logs/ and experiments/.

    Include suffixes when finding the largest existing run number.
    """
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)(?:_.*)?$")
    rid = max(
        [get_latest_run_id("tb_logs", base), get_latest_run_id("experiments", base)]
        + [
            int(m.group(1))
            for directory in ("tb_logs", "experiments")
            for entry in Path(directory).glob(f"{base}_*")
            if entry.is_dir() and (m := pattern.match(entry.name))
        ]
    ) + 1
    return f"{base}_{rid}"


def _write_initial_run_meta(
    exp_dir: Path,
    run_name: str,
    total_timesteps: int,
    resumed_from: dict | None = None,
) -> None:
    """Write settings needed to identify and reload a newly trained policy."""
    meta = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "total_timesteps": total_timesteps,
    }
    if resumed_from is not None:
        meta["resumed_from"] = resumed_from
    # Record the clearance-set settings needed to replay the checkpoint.
    meta.setdefault("action_set", str(Config.ACTION_SET))
    meta.setdefault("airport_mask", Config.AIRPORT_MASK)
    meta.setdefault("mask_frame_fix", bool(Config.MASK_FRAME_FIX))
    meta.setdefault("turn_mask_fix", bool(Config.TURN_MASK_FIX))
    (exp_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def _update_run_meta(exp_dir: Path, **updates) -> None:
    """Merge key/value updates into an existing run_meta.json."""
    meta_path = exp_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2))


def warmup_cosine_schedule(lr_max: float, lr_min: float, warmup_frac: float = 0.08):
    """Linear warm-up then half-cosine decay (SB3 `progress_remaining` 1->0 convention).

    Phase 1 (first `warmup_frac` of training): LR ramps linearly 0 -> `lr_max`. Starting from
    0 lets the critic build a baseline estimate before the actor takes
    large steps. Phase 2: half-period cosine from `lr_max` -> `lr_min` over the remainder, decaying
    slowly through mid-training and flattening into the floor near the end so late updates do not
    disrupt the converged autoregressive policy.

    Note: SB3 initialises the optimizer with `lr_schedule(1)` (== progress_remaining 1.0, i.e. the
    very start), which is 0.0 here; SB3 overwrites it via `_update_learning_rate` before the first
    gradient step, so the warm-up takes effect immediately.
    """

    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining  # 0.0 at start -> 1.0 at end
        if progress < warmup_frac:
            return lr_max * (progress / warmup_frac)
        decay = min(max((progress - warmup_frac) / (1.0 - warmup_frac), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay))
        return lr_min + (lr_max - lr_min) * cosine

    return schedule


def critic_warmup_schedule(after, total_timesteps: int, warmup_steps: int, critic_lr: float):
    """Constant ``critic_lr`` for the first ``warmup_steps`` timesteps, then ``after`` (an SB3
    schedule) re-parameterised over the REMAINING steps, so its own warm-up and cosine run in
    full once the policy is unfrozen."""
    total = float(total_timesteps)
    rest = max(1.0, total - float(warmup_steps))

    def schedule(progress_remaining: float) -> float:
        done = (1.0 - progress_remaining) * total
        # Tolerate floating-point rounding at the integer warm-up boundary.
        if done < warmup_steps - 0.5:
            return critic_lr
        return after(min(1.0, max(0.0, (total - done) / rest)))

    return schedule
