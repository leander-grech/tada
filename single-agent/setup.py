"""Module for installing the dependencies including the dynamic ones"""

import platform
import re
import sys
from pathlib import Path

from setuptools import setup


def _find_best_matching_wheel():
    simulator_dir = Path(__file__).parent / "simulator/wheels"
    wheels = list(simulator_dir.glob("flight_simulator-*.whl"))

    if not wheels:
        raise RuntimeError("No flight simulator wheels found in simulator directory")

    # Get current Python version and platform info
    py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    current_platform = platform.system().lower()
    current_machine = platform.machine()

    if current_platform == "darwin":
        current_platform = "macosx"
    elif current_platform == "windows":
        current_platform = "win"
        current_machine = "amd64"
    # Filter wheels based on Python version and platform
    matching_wheels = []
    for wheel in wheels:
        wheel_name = wheel.name
        if (
            py_version in wheel_name
            and current_platform in wheel_name.lower()
            and current_machine in wheel_name.lower()
        ):
            matching_wheels.append(wheel)

    if not matching_wheels:
        raise RuntimeError(
            f"No compatible wheel found for Python {py_version} \
                on {current_platform} {current_machine}. "
            f"Available wheels: {[w.name for w in wheels]}"
        )

    # Sort wheels by version number and take the latest
    def get_version(wheel_name):
        match = re.search(r"flight_simulator-(\d+\.\d+\.\d+)", wheel_name)
        return tuple(map(int, match.group(1).split("."))) if match else (0, 0, 0)

    best_wheel = max(matching_wheels, key=lambda w: get_version(w.name))
    return best_wheel


simulator_path = _find_best_matching_wheel()

setup(
    install_requires=[
        # Base runtime dependencies.
        # (training/plotting/eval extras live under the optional "train" group in
        #  pyproject.toml; install those with `pip install -e .[train]`)
        "stable-baselines3~=2.9.0",
        "torch~=2.8.0",
        "numpy>=2,<3",
        "gymnasium~=1.1.1",
        "python-dotenv~=1.0.1",
        f"flight_simulator @ file://{simulator_path.resolve()}",
    ]
)
