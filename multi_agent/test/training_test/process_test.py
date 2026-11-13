"""Behavioral tests for the flock run lock, process-group launch and stop."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from training import process

CHILD = {
    "sleep": "import time; time.sleep(600)",
    "ignore_term": (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"
    ),
    "cooperative": (
        "import time; from training import process\n"
        "while not process.stop_requested():\n"
        "    process.heartbeat('loop'); time.sleep(0.1)"
    ),
    "orphan_maker": (
        "import os, signal, subprocess\n"
        "subprocess.Popen(['sleep', '600'])\n"
        "os.kill(os.getpid(), signal.SIGKILL)"
    ),
}


def wait_for(predicate, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def spawn_serve(root, source, grace=2):
    command = [sys.executable, "-c", source]
    return subprocess.Popen(process.serve_argv(root, "t", grace, command), start_new_session=True)


def owner_pgid(root):
    assert wait_for(lambda: (root / "owner.json").exists())
    return process.read_json(root / "owner.json")["pgid"]


def test_second_acquire_is_refused_while_held(tmp_path):
    held = process.acquire(tmp_path)
    with pytest.raises(process.RunBusy):
        process.acquire(tmp_path)
    held.close()
    process.acquire(tmp_path).close()


def test_live_group_blocks_acquire_after_lock_holder_died(tmp_path):
    server = spawn_serve(tmp_path, CHILD["sleep"])
    pgid = owner_pgid(tmp_path)
    server.kill()  # flock released, child group still alive
    server.wait()
    assert process.members(pgid)
    with pytest.raises(process.RunBusy):
        process.acquire(tmp_path)
    process.stop(tmp_path, grace=1, term_wait=2)
    process.acquire(tmp_path).close()


def test_cooperative_stop_exits_without_signals(tmp_path):
    server = spawn_serve(tmp_path, CHILD["cooperative"])
    owner_pgid(tmp_path)
    assert wait_for(lambda: (tmp_path / "heartbeat.json").exists())
    process.stop(tmp_path, grace=15, term_wait=2)
    server.wait(timeout=20)
    assert process.read_json(tmp_path / "result.json")["returncode"] == 0
    assert not (tmp_path / "owner.json").exists()


def test_term_ignoring_child_is_killed(tmp_path):
    server = spawn_serve(tmp_path, CHILD["ignore_term"])
    pgid = owner_pgid(tmp_path)
    process.stop(tmp_path, grace=1, term_wait=1)
    server.wait(timeout=20)
    assert not process.members(pgid)
    assert process.read_json(tmp_path / "result.json")["returncode"] == -signal.SIGKILL


def test_stop_keeps_owner_while_group_survives(tmp_path, monkeypatch):
    process.atomic_json(tmp_path / "owner.json", {"pgid": 424242})
    monkeypatch.setattr(process, "members", lambda pgid: [pgid])
    monkeypatch.setattr(process, "terminate_group", lambda pgid, wait: None)
    assert process.stop(tmp_path, grace=0, term_wait=0) == "stuck"
    assert (tmp_path / "owner.json").exists()


def test_crashed_child_leaves_no_orphans(tmp_path):
    server = spawn_serve(tmp_path, CHILD["orphan_maker"], grace=1)
    pgid = owner_pgid(tmp_path)
    server.wait(timeout=30)
    assert not process.members(pgid)
    assert process.read_json(tmp_path / "result.json")["returncode"] == -signal.SIGKILL
    process.acquire(tmp_path).close()


def test_launch_refuses_while_lock_held(tmp_path):
    held = process.acquire(tmp_path)
    with pytest.raises(process.RunBusy):
        process.launch(tmp_path, [sys.executable, "-c", "pass"], wait=True)
    held.close()


def test_exec_returns_the_child_exit_code(tmp_path):
    command = [sys.executable, "-c", "import sys; sys.exit(3)"]
    assert process.launch(tmp_path, command, wait=True) == 3
    assert process.status(tmp_path)["state"] == "idle"


def test_exec_reports_the_child_signal_from_the_result(tmp_path):
    command = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"]
    assert process.launch(tmp_path, command, wait=True) == -signal.SIGKILL


def test_heartbeat_roundtrip_and_stale_status(tmp_path, monkeypatch):
    monkeypatch.setenv(process.RUN_DIR_ENV, str(tmp_path))
    process.heartbeat("training-iteration", iteration=3)
    beat = process.read_json(tmp_path / "heartbeat.json")
    assert beat["stage"] == "training-iteration" and beat["iteration"] == 3
    assert process.status(tmp_path)["state"] == "idle"
    server = spawn_serve(tmp_path, CHILD["sleep"])
    owner_pgid(tmp_path)
    assert process.status(tmp_path)["stale"] is False
    beat["wall"] -= process.STALE_HEARTBEAT_S + 1
    (tmp_path / "heartbeat.json").write_text(json.dumps(beat))
    assert process.status(tmp_path)["stale"] is True
    process.stop(tmp_path, grace=1, term_wait=2)
    server.wait(timeout=20)


def test_relative_root_reaches_children_that_change_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    source = (
        "import os; from training import process; os.chdir('elsewhere'); process.heartbeat('moved')"
    )
    command = [sys.executable, "-c", source]
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    assert process.launch(Path("slot"), command, wait=True) == 0
    assert process.read_json(tmp_path / "slot" / "heartbeat.json")["stage"] == "moved"


def test_run_dir_helpers_are_noops_when_unset(monkeypatch):
    monkeypatch.delenv(process.RUN_DIR_ENV, raising=False)
    process.heartbeat("x")
    process.request_stop("x")
    assert process.stop_requested() is False
