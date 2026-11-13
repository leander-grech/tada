"""Native artifact provenance and executable checks for the selected runtime."""

import argparse
import json
import os
import sys

from common.device import DEVICES
from common.native_runtime import STATE, install_native, output, verify_native
from common.runtime_log import runtime_print

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def model_probe(device):
    import gymnasium as gym
    import torch
    from ray.rllib.core.columns import Columns
    from torch_geometric.nn import GCNConv

    from atc_env.env_config import observation_settings_from_config
    from config.config import Config
    from config.settings import settings_from_config
    from models.observation_space import build_observation_space
    from network.rlm import ATCRLM, unbound_model_config

    target = torch.device(device)
    module = ATCRLM(
        observation_space=build_observation_space(observation_settings_from_config()),
        action_space=gym.spaces.Discrete(int(Config.NUM_ACTIONS)),
        model_config=unbound_model_config(settings_from_config().architecture),
    ).to(target)
    obs = {
        "aircraft_scalar": torch.randn(2, int(Config.AIRCRAFT_SCALAR_LEN), device=target),
        "global_context": torch.randn(2, int(Config.GLOBAL_CONTEXT_DIM), device=target),
        "flight_plans_rel": torch.randn(
            2, int(Config.MAX_WAYPOINTS), int(Config.FLIGHT_PLAN_WAYPOINT_DIM), device=target
        ),
        "flight_plan_mask": torch.ones(2, int(Config.MAX_WAYPOINTS), device=target),
        "neighbor_features": torch.randn(
            2, int(Config.MAX_NEIGHBORS), int(Config.NEIGHBOR_FEATURE_DIM), device=target
        ),
        "neighbor_mask": torch.ones(2, int(Config.MAX_NEIGHBORS), device=target),
    }
    batch = {
        Columns.OBS: {
            "action_mask": torch.ones(2, int(Config.NUM_ACTIONS), device=target),
            "observations": obs,
        }
    }
    result = module._forward_train(batch)
    values = module.compute_values(batch, embeddings=result[Columns.EMBEDDINGS])
    loss = result[Columns.ACTION_DIST_INPUTS].square().mean() + values.square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.gru.parameters()
    ):
        raise RuntimeError("ATCRLM forward/backward did not produce finite loss and GRU gradients")
    graph = GCNConv(4, 3).to(target)
    vertices = torch.randn(3, 4, device=target, requires_grad=True)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=target)
    graph(vertices, edges).square().mean().backward()
    if vertices.grad is None or not torch.isfinite(vertices.grad).all():
        raise RuntimeError("Graph backward failed")
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    return {"loss": float(loss.detach()), "gru_backward": True, "graph_backward": True}


def verify(device, expected_backend=None, identity_only=False):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import ray
    import torch
    import torch.version
    from threadpoolctl import threadpool_info

    backend = (STATE / "backend").read_text().strip()
    if expected_backend is not None and backend != expected_backend:
        raise RuntimeError(f"Expected backend {expected_backend}, got {backend}")
    actual_backend = "cuda" if torch.version.cuda is not None else "cpu"
    if actual_backend != backend:
        raise RuntimeError(f"Requested {backend}, imported {torch.__version__}")
    if any(os.environ.get(key) != "1" for key in THREAD_VARIABLES):
        raise RuntimeError("Launcher thread policy was not applied before imports")
    if torch.get_num_threads() != 1:
        raise RuntimeError("Torch effective thread count differs from launcher policy")
    native = verify_native()
    if identity_only:
        return
    probe = model_probe(device)
    pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"Effective thread pool differs from launcher policy: {pools}")
    from training.resource_profile import runtime_inventory

    report = {
        "physical_inventory": runtime_inventory(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_capability": torch.cuda.get_device_capability(device)
        if device.startswith("cuda")
        else None,
        "backend": backend,
        "device": device,
        "ray": ray.__version__,
        "native": native,
        "uv": output("uv", "--version"),
        "threads": {key: os.environ[key] for key in THREAD_VARIABLES},
        "torch_threads": torch.get_num_threads(),
        "threadpools": pools,
        "probe": probe,
    }
    runtime_print("Runtime", "verified", report=json.dumps(report, sort_keys=True))
    (STATE / f"report-{device.replace(':', '-')}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-native", action="store_true")
    parser.add_argument("--native-wheel")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], default="cpu")
    parser.add_argument("--expect-backend", choices=DEVICES)
    args = parser.parse_args()
    if args.install_native:
        install_native(args.native_wheel)
    else:
        verify(args.device, args.expect_backend, args.identity_only)


if __name__ == "__main__":
    main()
