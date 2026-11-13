# Selected evaluation

Use the selected inference bundles and exact native artifact identified in the [model list](../README.md#selected-models). Their manifests authenticate numeric weights, saved inference settings and compatibility metadata.

| Setting | Value |
|---|---|
| Aircraft | 20 |
| Supplied seeds | 100 per procedure, `config/contracts/supplied-evaluation-seeds.csv` |
| Grading | Inclusive 60-second arrival window with realized safety reported separately |
| Saved horizon | 128 decisions |
| Evaluation policy limit | 192 decisions using `policy_time_limit_steps` |
| TMB extra TTL | Off |
| Action shield | None |
| Policy attempts | Greedy index 0; stochastic indices 1–20 |
| Rollout search | K=8, M=1, greedy start |

Attempts share worlds; 2,000 sampled attempts represent 100 world clusters, not 2,000 independent scenarios. These seeds were used for model selection.

`evaluation/forms/` produces per-episode and per-aircraft records. Its warning-pair flag is distinct from the loss-of-separation end reason. Unknown initial timing values remain blank; do not replace them with measured zeros. Success, safety, timing, completeness and action counts must remain separate metrics.

Supplied/development access is governed by the benchmark reader. Held-out and replacement reservations remain protected. A shared seed or target-time match alone does not establish identical worlds across trainers.

The loader resolves included scenario snapshots by content hash. Keep native and semantic checks enabled when evaluating a model.

## Run evaluation

Use the commands in the [multi-agent guide](../README.md#evaluation) to materialize the selected cohorts and run each policy. Each output contains `results.json`.

For rollout search, run `python -m scripts.rollout_search --help`. Use K=8 with replanning every decision (M=1), and retain replay checks when exporting results. `scripts.export_evaluation_forms` converts replay records into episode and aircraft CSVs.

The single-agent `evaluation/seeds.txt` contains the same 100 seed values in the same order as the supplied multi-agent cohort.
