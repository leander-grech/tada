"""Retained wheel provenance and installed native-extension authentication."""

import email
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from common.hashing import file_sha256
from common.jsonio import read_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs/runtime"
DEFAULT_WHEEL = (
    ROOT / "simulator/wheels/flight_simulator-0.2.80-cp312-cp312-manylinux_2_34_x86_64.whl"
)
DEFAULT_WHEEL_SHA256 = "00880c2a3ece11eb78cc4d78c9feee7ad4c9ef656e038859ee11919dd075d9ec"


def output(*args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def install_manifest(manifest):
    wheel = Path(manifest["wheel"])
    if file_sha256(wheel) != manifest["wheel_sha256"]:
        raise RuntimeError("Retained native wheel checksum changed")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--reinstall",
            str(wheel),
        ],
        check=True,
    )
    subprocess.run(["uv", "pip", "check", "--python", sys.executable], check=True)
    (STATE / "native.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_native()


def wheel_metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise RuntimeError("Expected one wheel package metadata record")
        metadata = email.message_from_bytes(archive.read(names[0]))
        if metadata["Name"].lower().replace("_", "-") != "flight-simulator":
            raise RuntimeError("Supplied wheel is not flight-simulator")
        return {
            "version": metadata["Version"],
            "requires_python": metadata["Requires-Python"],
            "requires_dist": metadata.get_all("Requires-Dist", []),
            "generator": archive.read(names[0].replace("METADATA", "WHEEL")).decode(),
        }


def supplied_manifest(path):
    supplied = Path(path).resolve()
    supplied_hash = file_sha256(supplied)
    directory = STATE / "external" / supplied_hash
    directory.mkdir(parents=True, exist_ok=True)
    retained = directory / supplied.name
    if retained != supplied:
        shutil.copyfile(supplied, retained)
    if supplied.suffix == ".zip":
        with zipfile.ZipFile(retained) as archive:
            members = archive.infolist()
            if (
                len(members) != 1
                or not members[0].filename.endswith(".whl")
                or Path(members[0].filename).name != members[0].filename
            ):
                raise RuntimeError("Supplied archive must contain exactly one wheel at its root")
            wheel = directory / members[0].filename
            wheel.write_bytes(archive.read(members[0]))
    elif supplied.suffix == ".whl":
        wheel = retained
    else:
        raise RuntimeError("Supplied artifact must be ZIP or WHL")
    metadata = wheel_metadata(wheel)
    manifest = {
        "mode": "external",
        "source_provenance": "unverified external artifact; not attributed to checked-in Rust source",
        "supplied_artifact": str(retained),
        "supplied_sha256": supplied_hash,
        "wheel": str(wheel),
        "wheel_sha256": file_sha256(wheel),
        "package_metadata": metadata,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def install_native(supplied=None):
    previous = STATE / "native.json"
    if supplied is None and previous.exists():
        selected = read_json(previous)
        if selected.get("mode") == "external":
            retained = Path(selected["supplied_artifact"])
            if file_sha256(retained) != selected["supplied_sha256"]:
                raise RuntimeError("Retained supplied artifact checksum changed")
            # Reuse the original identity; never silently bless changed wheel bytes.
            install_manifest(selected)
            return
    if supplied == "source":
        raise ValueError("Source builds are not supported; supply a simulator wheel or ZIP")
    if supplied is None and file_sha256(DEFAULT_WHEEL) != DEFAULT_WHEEL_SHA256:
        raise RuntimeError("Included simulator wheel checksum changed")
    install_manifest(supplied_manifest(DEFAULT_WHEEL if supplied is None else supplied))


def verify_native():
    manifest = read_json(STATE / "native.json")
    identity = None
    mode = manifest.get("mode")
    if mode == "external":
        retained = Path(manifest["supplied_artifact"])
        if file_sha256(retained) != manifest["supplied_sha256"]:
            raise RuntimeError("Retained supplied artifact checksum changed")
    else:
        raise RuntimeError("Unknown native provenance mode")
    wheel = Path(manifest["wheel"])
    if file_sha256(wheel) != manifest["wheel_sha256"]:
        raise RuntimeError("Retained native wheel checksum changed")
    metadata = wheel_metadata(wheel)
    if importlib.metadata.version("flight-simulator") != metadata["version"]:
        raise RuntimeError("Installed native version differs from retained wheel")
    package = importlib.import_module("flight_simulator")
    extension = package
    if not str(package.__file__).endswith(".so"):
        extension = importlib.import_module("flight_simulator.flight_simulator")
    if extension.__file__ is None:
        raise RuntimeError("Native extension has no file location")
    location = Path(extension.__file__).resolve()
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Native extension imported outside selected environment: {location}")
    with zipfile.ZipFile(wheel) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.endswith("/" + location.name) or name == location.name
        ]
        if len(matches) != 1 or archive.read(matches[0]) != location.read_bytes():
            raise RuntimeError("Imported extension bytes do not match retained wheel")
    return {
        "mode": mode,
        "version": metadata["version"],
        "wheel_metadata": metadata["generator"],
        "source_identity": identity,
        "source_provenance": manifest.get("source_provenance", "unverified external artifact"),
        "wheel_sha256": manifest["wheel_sha256"],
        "extension": str(location),
        "extension_sha256": file_sha256(location),
    }
