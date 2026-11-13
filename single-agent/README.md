# Single-agent ATC policy

JAX PPO with a windowed, autoregressive controller for MXP trombone (TMB) and BGY point-merge (PMS) arrival control. Policies load through Stable-Baselines3 for evaluation.

## Setup

Use CPython 3.12 on Linux x86_64 with glibc 2.34 or newer. Shared simulator builds are in [`artifacts/simulator/wheels/`](../artifacts/simulator/wheels/); see [platform selection](../artifacts/simulator/README.md). From `single-agent/`, activate your Python environment, copy the Linux wheel into `simulator/wheels/` and install the CPU dependencies:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
sed 's/^jax\[cuda12\]/jax/' recipes/requirements-windowed.txt | python -m pip install -r /dev/stdin
python -m pip install simulator/wheels/flight_simulator-0.2.80-cp312-cp312-manylinux_2_34_x86_64.whl
```

Aircraft performance data is not distributed: supply an A320 design as described in [aircraft performance data](../docs/aircraft-designs.md) first. For CUDA training, use the unchanged requirements file instead of the CPU substitution; see [dependency setup](recipes/README.md). Run tests with `TADA_SEQUENCE_OBS=1 TADA_MAX_PICKS=2 python -m pytest tests` (the published models use this multi-pick configuration; `test_wheel_selection` also expects the manylinux wheel in `simulator/wheels/`). Checkpoint-dependent numerical checks are executable scripts under `tests/`.

## Selected models

| Procedure | Bundle in `artifacts/single_agent/` | Action set | Evaluation mask |
|---|---|---|---|
| MXP (TMB) | `mxp/` | `v2` | `procedure` |
| PMS (BGY) | `pms/` | `v2pms` | none |

Keep each bundle’s `final_model.zip`, `run_meta.json` and `vecnormalize.pkl` together. Both policies use sequence observations, two picks per interval and the corrected mask frame. From the repository root:

```sh
(cd artifacts/single_agent && sha256sum -c SHA256SUMS)
```

## Evaluation

From `single-agent/`, evaluate the supplied 100-seed cohort with 20 aircraft and rollout search K=8, M=10:

```sh
export CUDA_VISIBLE_DEVICES= TADA_GENERATOR=procedure
export TADA_SEQUENCE_OBS=1 TADA_MAX_PICKS=2 TADA_MASK_FRAME_FIX=1
TADA_ACTION_SET=v2 TADA_AIRPORT_MASK=procedure \
  python -m evaluation.run_search --procedure tmb \
  --model ../artifacts/single_agent/mxp/final_model.zip \
  --seed-file evaluation/seeds.txt --out runs/mxp-search
TADA_ACTION_SET=v2pms TADA_AIRPORT_MASK= \
  python -m evaluation.run_search --procedure pms \
  --model ../artifacts/single_agent/pms/final_model.zip \
  --seed-file evaluation/seeds.txt --out runs/pms-search
```

These commands run greedy-initialized attempt 0 plus sampled attempts 1–20; use `--attempts 0` for a single attempt and `--workers` to set process capacity. Resume an interrupted run with the same inputs and `--resume`. Output includes episode and aircraft CSVs; `--picks` adds executed decisions. Required model and procedure settings are checked before evaluation.

Use `python -m analysis.score_windowed --help` for policy-only scoring, and [evaluation tools](evaluation/README.md) for CSV export. Random extra TTL is disabled. The supplied evaluation seeds were used for model selection; success, safety and timing remain separate metrics.

## Training

Use `main_jax.py` for fresh training. With the shared environment settings above, an MXP example is:

```sh
TADA_ACTION_SET=v2 TADA_AIRPORT_MASK=procedure \
  python main_jax.py --env windowed --reward-mode outcome_pbrs \
  --use-case 1 --flights 20 --seed 42 \
  --train-seed-pool config/training_seeds/training_seeds.txt \
  --n-envs 4 --total-timesteps 1000000 --run-suffix mxp
```

For PMS, select `TADA_ACTION_SET=v2pms`, clear `TADA_AIRPORT_MASK`, use `--use-case 2` and the same `config/training_seeds/training_seeds.txt` pool. Training writes models and settings under `experiments/` and TensorBoard output under `tb_logs/`. Choose worker count and training budget for your hardware; see `python main_jax.py --help` for optimizer and warm-start options.

## Source layout

- `atc_env/`, `actions/`, `models/`, `network/`: environment, observations and policy.
- `jax_ppo/`, `main_jax.py`: JAX learner and training entry point.
- `evaluation/`: search runner, supplied seeds and CSV export.
- `analysis/`: scoring, rollout search and episode recording.
- `config/`, `static/`, `simulator/`: required inputs, Python adapters and native wheel.
- `tests/`: behavior, compatibility, resume and numerical checks.
