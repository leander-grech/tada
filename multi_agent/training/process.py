"""Single-run safety: flock lock, setsid launch, stop.json, heartbeat, killpg escalation.

`serve` is a session leader that holds the run lock and starts the command in its
own process group; every exit path of `serve` cleans that group. `stop` asks
cooperatively through `stop.json`, then escalates with killpg TERM and KILL.
Descendants that call setsid (or setpgid) leave the group and escape killpg.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from common.atomic_io import atomic_json
from common.runtime_log import runtime_print

DEFAULT_ROOT = Path("runs/run-slot")
RUN_DIR_ENV = "TADA_RUN_DIR"
STALE_HEARTBEAT_S = 900
POLL_S = 0.2
_COMPONENT = "Process"


class RunBusy(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def run_dir():
    value = os.environ.get(RUN_DIR_ENV)
    return Path(value) if value else None


def heartbeat(stage, **values):
    root = run_dir()
    if root:
        atomic_json(root / "heartbeat.json", {"stage": stage, "wall": time.time(), **values})


def stop_requested():
    root = run_dir()
    return bool(root and (root / "stop.json").exists())


def request_stop(reason):
    root = run_dir()
    if root:
        atomic_json(root / "stop.json", {"reason": reason, "wall": time.time()})


def members(pgid):
    """Live (non-zombie) pids in a process group."""
    listing = subprocess.run(
        ["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True, check=True
    ).stdout
    found = []
    for line in listing.splitlines():
        pid, group, state = line.split()
        if int(group) == pgid and not state.startswith("Z"):
            found.append(int(pid))
    return found


def _wait_empty(pgid, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not members(pgid):
            return True
        time.sleep(POLL_S)
    return not members(pgid)


def terminate_group(pgid, wait):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not members(pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        _wait_empty(pgid, wait)


def acquire(root):
    """Take the non-blocking run lock; refuse while a previous group still lives."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "run.lock").open("w")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RunBusy(f"another run holds {root / 'run.lock'}") from None
    owner = root / "owner.json"
    if owner.exists() and members(read_json(owner)["pgid"]):
        stream.close()
        raise RunBusy(f"previous process group {read_json(owner)['pgid']} is still alive; run stop")
    for name in ("owner.json", "stop.json", "heartbeat.json", "result.json"):
        (root / name).unlink(missing_ok=True)
    return stream


def serve(root, token, command, grace):
    """Session leader for one run; its command's process group never outlives it."""
    lock = acquire(root)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    # Absolute: Tune runs trainables from per-trial working directories.
    env = {
        **os.environ,
        RUN_DIR_ENV: str(Path(root).resolve()),
        "RAY_ADDRESS": "local",
        "PYTHONHASHSEED": "0",
    }
    child = subprocess.Popen(command, env=env, process_group=0)
    atomic_json(
        root / "owner.json",
        {
            "token": token,
            "pid": os.getpid(),
            "pgid": child.pid,
            "command": command,
            "started": time.time(),
            "log": f"serve-{token}.log",
        },
    )
    returncode = child.wait()
    atomic_json(
        root / "result.json", {"token": token, "returncode": returncode, "wall": time.time()}
    )
    terminate_group(child.pid, grace)
    (root / "owner.json").unlink(missing_ok=True)
    lock.close()
    return returncode


def serve_argv(root, token, grace, command):
    return [
        sys.executable, "-m", "training.process", "--root", str(root),
        "_serve", "--token", token, "--grace", str(grace), "--", *command,
    ]  # fmt: skip


def _published(path, token):
    return path.exists() and read_json(path)["token"] == token


def launch(root, command, wait, grace=30):
    """Start `serve` in a new session; optionally block until the run ends."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    log = root / f"serve-{token}.log"
    with log.open("w") as stream:
        server = subprocess.Popen(
            serve_argv(root, token, grace, command),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.time() + 15
    while not (_published(root / "owner.json", token) or _published(root / "result.json", token)):
        if server.poll() is not None:
            raise RunBusy(log.read_text()[-2000:])
        if time.time() > deadline:
            raise RuntimeError(f"run did not start; see {log}")
        time.sleep(POLL_S)
    runtime_print(_COMPONENT, "started", token=token, log=log)
    if not wait:
        return 0
    try:
        server.wait()
    except KeyboardInterrupt:
        stop(root)
        server.wait()
    sys.stdout.write(log.read_text())
    return read_json(root / "result.json")["returncode"]


def stop(root, grace=30, term_wait=10):
    root = Path(root)
    owner = root / "owner.json"
    if not owner.exists():
        return "idle"
    pgid = read_json(owner)["pgid"]
    atomic_json(root / "stop.json", {"reason": "operator", "wall": time.time()})
    if not _wait_empty(pgid, grace):
        terminate_group(pgid, term_wait)
    if members(pgid):
        return "stuck"
    owner.unlink(missing_ok=True)
    return "stopped"


def status(root):
    root = Path(root)
    owner = root / "owner.json"
    beat = root / "heartbeat.json"
    report = {"state": "idle", "stale": False}
    if owner.exists():
        info = read_json(owner)
        state = "running" if members(info["pgid"]) else "stale-owner"
        report.update(state=state, pgid=info["pgid"], command=info["command"])
    if beat.exists():
        last = read_json(beat)
        age = time.time() - last["wall"]
        report.update(
            heartbeat=last,
            heartbeat_age_s=round(age),
            stale=report["state"] == "running" and age > STALE_HEARTBEAT_S,
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="training.process")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("launch", "exec", "_serve"):
        item = sub.add_parser(name)
        item.add_argument("--grace", type=int, default=30)
        if name == "_serve":
            item.add_argument("--token", required=True)
        item.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    sub.add_parser("stop")
    args = parser.parse_args(argv)
    if args.action in ("launch", "exec", "_serve"):
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if args.action == "_serve":
            return serve(args.root, args.token, command, args.grace)
        return launch(args.root, command, wait=args.action == "exec", grace=args.grace)
    if args.action == "status":
        sys.stdout.write(json.dumps(status(args.root), indent=2) + "\n")
        return 0
    outcome = stop(args.root)
    sys.stdout.write(outcome + "\n")
    return int(outcome == "stuck")


if __name__ == "__main__":
    sys.exit(main())
