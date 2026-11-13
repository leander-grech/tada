# Run configuration and protected inputs

Resolved JSON contracts configure training and evaluation without mutating global settings. An overlay uses `base` and `patch`; relative paths resolve from the declaring file.

`scenario.use_case` selects the MXP or BGY scenario snapshot. Native physics, preprocessing, scenario hashes and reserved identities are checked before bound generation. `reserved-mxp-v1.json`, versioned conditions and the BGY inputs under `bgy/` remain required by runtime guards and test fixtures. Keep these inputs unchanged, including the predecessors referenced by hash-linked conditions. The `commit` and `note` fields in the conditions files record internal development provenance at freeze time and stay unchanged because the files are hash-linked.

`supplied-evaluation-seeds.csv` is the selected development cohort. The benchmark reader controls access; held-out and replacement identities must remain excluded from training and ordinary operational streams.

For selected-model inference use the selected inference bundle and its resolved run settings. The effective saved contract defines the model’s evaluation settings.
