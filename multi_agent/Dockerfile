# Build from multi_agent; setup resolves package specs unless a lockfile is supplied.
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
ENV CUDA_VISIBLE_DEVICES="" RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
RUN sh scripts/runtime.sh setup --backend cpu
ENV PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
    PYTHONHASHSEED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_TMPDIR=/tmp/tada-ray \
    CHECKPOINTS_DIR=/app/checkpoints
RUN python -c 'from common.native_runtime import verify_native; verify_native(); import inference.server, training.checkpoints, atc_env.completed_scenario'
EXPOSE 8000
ENTRYPOINT ["python", "-m", "inference.server"]
