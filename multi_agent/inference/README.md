# Inference

The inference service reads the selected inference bundles identified by `manifest.json`. Required payloads are authenticated by that manifest; token and native-artifact checks preserve model/environment semantics.

Use the selected TMB and PMS bundles described in the [model list](../README.md#selected-models). The registered service entry point is `inference-server`; its module is `inference.server`. Model directory configuration uses `CHECKPOINTS_DIR`. The loader locates included scenario snapshots by content hash.

Selected inference bundles use numeric `weights.npz` with pickle disabled. The reader also supports complete training checkpoints, which deserialize Python pickle and must come from a trusted source.

Candidate selection uses canonical evidence eligibility and safety-aware outcome ranking. Report safety, timing and command burden separately. Serve stochastic requests sequentially for reproducible sampling because the simulator uses process-global random state.

## Run the service

From the repository root, build the CPU image and mount each model as an immediate child of its airport directory:

```sh
docker build -t tada-multi-agent multi_agent
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts/multi_agent/mxp:/app/checkpoints/mxp/tmb:ro" \
  -v "$(pwd)/artifacts/multi_agent/pms:/app/checkpoints/bgy/pms:ro" \
  tada-multi-agent
```

The server discovers bundles under `CHECKPOINTS_DIR/mxp/<name>/` and `CHECKPOINTS_DIR/bgy/<name>/` and loads a default for each airport. The image sets `CHECKPOINTS_DIR=/app/checkpoints`. Query `http://localhost:8000/health` for active models or open `http://localhost:8000/docs` for the API.

The server binds to `0.0.0.0` by default (`INFERENCE_HOST`), and API-key authentication is disabled by default. Set `INFERENCE_API_KEY_ENABLED=true` and `INFERENCE_API_KEY` before exposing the server beyond a trusted network; clients then send the key in the `X-API-Key` header.
