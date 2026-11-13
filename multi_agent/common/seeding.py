"""Process-wide RNG seeding for reproducible evaluation and training."""

import os
import random
import sys

import numpy as np
import torch

SINGLE_THREAD_BLAS_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def seed_all(seed: int, *, deterministic: bool = False) -> None:
    """Seed all RNGs for reproducible evaluation/training.

    Args:
        seed: RNG seed for all backends.
        deterministic: If True, re-exec with ``PYTHONHASHSEED`` and
            single-threaded BLAS env vars, force single-threaded torch,
            and enable deterministic algorithms.
            Use for offline evaluation; too restrictive for training.
    """
    if deterministic:
        targets = {var: "1" for var in SINGLE_THREAD_BLAS_ENV_VARS}
        targets["PYTHONHASHSEED"] = str(seed)
        need_reexec = any(os.environ.get(k) != v for k, v in targets.items())
        os.environ.update(targets)
        if need_reexec:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
