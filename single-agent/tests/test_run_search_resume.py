"""Resume must preserve complete results and reject incompatible runs."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("run_search_test", ROOT / "evaluation/run_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import actions.action_set as actions
    monkeypatch.setattr(actions, "sync_obs_width_to_checkpoint", lambda _: None)
    monkeypatch.setattr(mod.pr, "category_of", lambda _: None)
    model = tmp_path / "final_model.zip"
    model.write_bytes(b"model one")
    (tmp_path / "vecnormalize.pkl").write_bytes(b"normalization")
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("101 102\n")
    out = tmp_path / "out"
    class Pool:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def imap_unordered(self, func, jobs, chunksize):
            for seed, attempt in jobs:
                ep = dict(seed=seed, attempt=attempt, n_flights=2, n_picks=2, solved=True,
                          sep=False, on_time=1.0, wall_s=0)
                ac = [dict(seed=seed, attempt=attempt, aircraft_index=i, clearances=12345) for i in range(2)]
                picks = [dict(seed=seed, attempt=attempt, aircraft_index=i, pick=i) for i in range(2)]
                yield ep, ac, picks
    monkeypatch.setattr(mod, "Pool", Pool)
    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["run_search.py", "--model", str(model), "--procedure", "tmb",
                                        "--out", str(out), "--seed-file", str(seed_file), "--seeds", "2",
                                        "--attempts", "0", "--flights", "2", "--K", "5", "--picks", *extra])
        mod.main()
    return run, out, model


def rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_complete_resume_skips_and_preserves_outputs(runner):
    run, out, _ = runner
    run()
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    run("--resume")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


@pytest.mark.parametrize("damaged", ["tmb_aircraft.csv", "tmb_picks.csv"])
def test_resume_repairs_incomplete_aircraft_and_pick_rows(runner, damaged):
    run, out, _ = runner
    run()
    path = out / damaged
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:2] + lines[3:]))
    run("--resume")
    for name, index in (("tmb_aircraft.csv", "aircraft_index"), ("tmb_picks.csv", "pick")):
        data = rows(out / name)
        assert sorted(int(r[index]) for r in data if r["seed"] == "101") == [0, 1]
        assert len(data) == 4
    assert len(rows(out / "tmb_episodes.csv")) == 2


@pytest.mark.parametrize("changed", ["model", "K"])
def test_resume_rejects_changed_settings_without_writes(runner, changed):
    run, out, model = runner
    run()
    if changed == "model":
        model.write_bytes(b"different model")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(SystemExit):
        run("--resume", *( ["--K", "3"] if changed == "K" else []))
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_resume_recomputes_unterminated_aircraft_record(runner):
    run, out, _ = runner
    run()
    path = out / "tmb_aircraft.csv"
    path.write_bytes(path.read_bytes()[:-3])
    run("--resume")
    data = rows(path)
    assert len(data) == 4
    assert all(r["clearances"] == "12345" for r in data)
    assert path.read_bytes().endswith(b"\n")


def test_resume_recovers_torn_aircraft_header(runner):
    run, out, _ = runner
    run()
    path = out / "tmb_aircraft.csv"
    path.write_bytes(b"seed,atte")
    run("--resume")
    data = rows(path)
    assert len(data) == 4
    assert all(r["clearances"] == "12345" for r in data)
