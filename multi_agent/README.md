# Multi-agent ATC policy

Shared-policy, recurrent PPO for MXP trombone (TMB) and BGY point-merge (PMS) arrival control, implemented with Ray RLlib and PyTorch.

## Setup

Use CPython 3.12 on Linux x86_64 with glibc 2.34 or newer, and install `uv`. Shared simulator builds are in [`artifacts/simulator/wheels/`](../artifacts/simulator/wheels/); see [platform selection](../artifacts/simulator/README.md). From `multi_agent/`, copy the Linux wheel into `simulator/wheels/` and set up CPU evaluation:

```sh
export CUDA_VISIBLE_DEVICES=
sh scripts/runtime.sh setup --backend cpu
sh scripts/runtime.sh runtime-check --device cpu
```

Aircraft performance data is not distributed: supply an A320 design as described in [aircraft performance data](../docs/aircraft-designs.md) first. Setup resolves the dependencies in `pyproject.toml`, uses `uv.lock` when supplied, and installs the included simulator wheel. Native checks authenticate the imported extension against that wheel. Run tests with `sh scripts/runtime.sh test`; use `check` for formatting, lint and types.

## Selected models

| Procedure | Bundle in `artifacts/multi_agent/` |
|---|---|
| MXP (TMB) | `mxp/` |
| PMS (BGY) | `pms/` |

Each inference bundle contains `manifest.json` and `weights.npz`; keep them together. The loader verifies weights, architecture, scenario inputs and compatibility settings. From the repository root:

```sh
(cd artifacts/multi_agent && sha256sum -c SHA256SUMS)
```

## Evaluation

The selected protocol uses 20 aircraft, 100 supplied seeds per procedure, inclusive 60-second arrival grading and a 192-decision evaluation limit. MXP random extra TTL is disabled. The supplied seeds were used for model selection.

From `multi_agent/`, create the cohorts and run greedy evaluation. Copies under `runs/` keep generated scenario records alongside the evaluation output:

```sh
export CUDA_VISIBLE_DEVICES=
mkdir -p runs/models
cp -R ../artifacts/multi_agent/mxp runs/models/mxp
cp -R ../artifacts/multi_agent/pms runs/models/pms
.venv/bin/python -m scripts.materialize_supplied --output runs/cohort \
  --procedure mxp --aircraft 20 --generator-variant mxp-no-random-extra-ttl
.venv/bin/python -m scripts.materialize_supplied --output runs/cohort \
  --procedure bgy --aircraft 20
.venv/bin/python -m evaluation.benchmark.evaluate --checkpoint runs/models/mxp \
  --manifest runs/cohort/mxp-development-20.json --output runs/mxp-greedy \
  --policy-time-limit-scale 1.5
.venv/bin/python -m evaluation.benchmark.evaluate --checkpoint runs/models/pms \
  --manifest runs/cohort/bgy-development-20.json --output runs/pms-greedy \
  --policy-time-limit-scale 1.5
```

Use fresh output directories. Each contains `results.json`; `--stochastic --k 20` evaluates sampled attempts. Selected rollout search uses K=8 and M=1. See the [evaluation guide](docs/evaluation.md) for search, replay and CSV export, and the [inference guide](inference/README.md) for serving.

## Training

Training uses resolved JSON configurations with protected seed reservations. The supplied training configurations select a CUDA learner. Configure the CUDA dependencies and an example 100-iteration MXP run from `multi_agent/`:

```sh
unset CUDA_VISIBLE_DEVICES
sh scripts/runtime.sh setup --backend cuda
mkdir -p runs
cat > runs/train-mxp.json <<'JSON'
{
  "base": "../config/contracts/mxp-base.json",
  "patch": {
    "scenario": {"aircraft_count": 20, "generator_variant": "mxp-no-random-extra-ttl"},
    "execution": {"output_root": ".", "stop_iterations": 100}
  }
}
JSON
sh scripts/runtime.sh run python -m main train \
  --mode fresh --config runs/train-mxp.json
```

For PMS, use `config/contracts/bgy/base.json` as the base and omit `generator_variant`. Choose resource and training budgets in the configuration. `status` and `stop` manage the launched process. See [run configuration](config/contracts/README.md) and `python -m main train --help` for resume and phase options.

## Source layout

- `atc_env/`, `actions/`, `models/`, `network/`: environment, observations and policy.
- `outcomes/`: safety, timing and completion evidence.
- `training/`: configuration, learners and checkpoint handling; also required for inference.
- `evaluation/`, `scripts/`, `inference/`: scoring, search, replay, export and serving.
- `config/`, `static/`, `simulator/`: required inputs, Python adapters and native wheel.
- `test/`, `inference/test/`: behavioral and integration tests.
