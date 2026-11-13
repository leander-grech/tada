# Aircraft performance data (prerequisite)

Aircraft performance data is licensed and is not part of this repository. Before running tests,
training, evaluation or inference, supply it as `A320.json` in a `designs/` directory under each
scenario configuration of the implementation you use:

```sh
for root in multi_agent/config/simulation_config_bgy multi_agent/config/simulation_config_mxp \
            single-agent/config/simulation_config_bgy single-agent/config/simulation_config_mxp; do
  mkdir -p "$root/designs" && cp /path/to/A320.json "$root/designs/"
done
```

The published models were trained and evaluated with Airbus A320-231 performance values derived
from EUROCONTROL BADA (Base of Aircraft Data). That file has SHA-256
`466a0686dd1e181625fbef570c7dc0b433e0fdbc9f68188c0ffae0d63d850fcc`. The multi-agent run and
benchmark manifests hash every file in the scenario configuration, so they accept only that exact
file, and `designs/` must contain nothing else. Other values change the results.

The simulator loads every `*.json` file in `designs/` as one aircraft design; the scenario
generator uses the ICAO type `A320`. Inference derives its aircraft designs from the same files.
All fields are required numbers unless shown as strings:

```json
{
  "icao_typename": "A320",
  "descriptive_name": "AIRBUS A320-231",
  "masses": {"minimum_kg": 0.0, "reference_kg": 0.0, "maximum_kg": 0.0},
  "limits": {
    "max_longitudinal_acceleration_knots_per_sec": 0.0,
    "max_op_altitude_ft": 0.0,
    "v_stall_flap_kcas": 0.0,
    "v_stall_clean_kcas": 0.0,
    "v_mo_kcas": 0.0,
    "max_climb_rate_fpm": 0.0,
    "max_descent_rate_fpm": 0.0
  },
  "performance_sectors": [
    {
      "altitude_min_ft": 0.0,
      "altitude_max_ft": 2000.0,
      "cruise": {"v_opt_kcas": 0.0},
      "climb": {"v_opt_kcas": 0.0, "roc_fpm": 0.0},
      "descent": {"v_opt_kcas": 0.0, "rod_fpm": 0.0}
    }
  ]
}
```

`performance_sectors` lists altitude bands in ascending order. The published models used five
bands: 0-2000, 3000-8000, 10000-16000, 18000-24000 and 26000-33000 ft.
