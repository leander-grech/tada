"""Release of locked flights, pay at release, and the flat deviation potential (windowed env).

  1. release_locked + pay_at_release, do-nothing: flights are released only from the front of the
     queue, released flights leave the window, each was locked (within release_dev_s, no predicted
     conflict), and the landing reward paid equals the objective exactly: landing_value(realised)
     for every landed flight, with no touchdown correction (a do-nothing prediction is exact).
  2. The shaping still telescopes with all three switches on, under random clearances.
  3. snapshot/restore (the lookahead) replays exactly with release on.
  4. Evaluation builds the env the run trained in: env_config_for / reward_config_for read run_meta.
  5. train_max_early_s: training streams are redrawn until feasible, deterministically per train_seed.
  6. Whole streams (10 and 2x20 flights) run to their end with release on: a selectable flight at every
     step, including the stream's tail where every airborne flight is released.

Run: TADA_SEQUENCE_OBS=1 TADA_MAX_PICKS=2 OMP_NUM_THREADS=1 python tests/test_release.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from atc_env.windowed_env import WindowedATCEnv, landing_value
from config.config import Config
from config.windowed_config import WindowedConfig

GAMMA = float(Config.PBRS_GAMMA)
HARD = WindowedConfig.HARD_BRACKETS
DO_NOTHING = 0
def run_checks():
    failures = []


    def check(name, ok, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)


    def release_env(**extra):
        return WindowedATCEnv({"evaluation_mode": True, "reward_mode": "outcome_pbrs", "landing_brackets": HARD,
                               "release_locked": True, "pay_at_release": True, **extra})


    def first_selectable(obs):
        return int(np.flatnonzero(obs["mask_select"])[0])


    # 1. do-nothing episodes on validation seeds until one releases flights
    print("1. release rule, do-nothing")
    env = release_env()
    for seed in [int(s) for s in Config.SCENARIO_EVAL_SEEDS][:12]:
        np.random.seed(seed % 2**32)
        obs, _ = env.reset(seed=seed, options={"scenario_seed": seed})
        order_ok = window_ok = True
        done, steps, info = False, 0, {}
        while not done and steps < 300:
            before = set(env._released)
            obs, r, te, tr, info = env.step(np.array([first_selectable(obs), DO_NOTHING, 0]))
            done, steps = te or tr, steps + 1
            for cs in set(env._released) - before:
                ahead = [c for c in env._eta if env._eta[c] < env._eta[cs]]
                order_ok &= all(c in env._landing_devs or c in env._released for c in ahead)
                order_ok &= abs(env._released[cs]) <= 30.0
            window_ok &= not ({ac.call_sign for ac in env._window} & set(env._released))
        if env._released:
            break
    m = info.get("episode_metrics") or {}
    check("some flights are released", len(env._released) > 0, f"seed {seed}: {len(env._released)} released, "
          f"{len(env._landing_devs)} landed")
    check("released only from the front of the queue, each within 30 s at release", order_ok)
    check("released flights leave the window", window_ok)
    check("no touchdown correction under do-nothing (the prediction holds)", m.get("window_release_correction", -1) == 0.0,
          f"correction {m.get('window_release_correction')}")
    busted = m.get("end_reason") in ("separation", "airspace_exit")
    unlanded_paid = sum(v for c, v in env._release_paid.items() if c not in env._landing_devs)
    # a bust takes back the release payments of flights that never landed; a truncation keeps them
    objective = sum(landing_value(abs(d), HARD) for d in env._landing_devs.values()) + (0.0 if busted else unlanded_paid)
    check("landing reward paid == the objective (bracket of each realised landing; advances returned on a bust)",
          abs(env._landing_bonus_total - objective) < 1e-9,
          f"{env._landing_bonus_total:.3f} vs {objective:.3f}, end {m.get('end_reason')}, {len(env._released) - sum(c in env._landing_devs for c in env._released)} released unlanded")

    # a bust with released flights still airborne: their release payments are taken back
    env = release_env(release_dev_s=1e9)
    found = False
    for seed in [int(s) for s in Config.SCENARIO_EVAL_SEEDS][:20]:
        np.random.seed(seed % 2**32)
        obs, _ = env.reset(seed=seed, options={"scenario_seed": seed})
        done, info = False, {}
        while not done:
            obs, r, te, tr, info = env.step(np.array([first_selectable(obs), DO_NOTHING, 0]))
            done = te or tr
        m = info.get("episode_metrics") or {}
        airborne_released = [c for c in env._released if c not in env._landing_devs]
        if m.get("end_reason") == "separation" and airborne_released:
            found = True
            break
    landed_value = sum(landing_value(abs(d), HARD) for d in env._landing_devs.values())
    check("a bust returns the release payments of flights still airborne",
          found and abs(env._landing_bonus_total - landed_value) < 1e-9,
          f"seed {seed}: {len(airborne_released) if found else 0} released airborne at the bust, "
          f"paid {env._landing_bonus_total:.3f} vs landed brackets {landed_value:.3f}")

    # 2. telescoping, all three switches, random clearances on stitched training streams
    print("2. shaping telescopes")
    env = WindowedATCEnv({"evaluation_mode": False, "reward_mode": "outcome_pbrs", "stitch_segments": 2,
                          "stitch_gap_s": (120.0, 900.0), "train_seed": 7, "landing_brackets": HARD,
                          "release_locked": True, "pay_at_release": True, "pbrs_flat_deviation": True})
    obs, _ = env.reset()
    phi0 = sum(env._potential())
    rng = np.random.default_rng(3)
    acc, g, done, steps = 0.0, 1.0, False, 0
    while not done and steps < 80:
        a = int(rng.choice(np.flatnonzero(obs["mask_select"])))
        valid = np.flatnonzero(obs["mask_action_per_ac"][a])
        c = DO_NOTHING if rng.random() < 0.6 else int(rng.choice(valid))
        obs, r, te, tr, info = env.step(np.array([a, c, int(rng.random() < 0.2)]))
        p = info["reward_parts"]
        acc += g * (p.deviation + p.conflict + p.goal)
        g *= 1.0 if info.get("zero_time") else GAMMA
        done, steps = te or tr, steps + 1
    target = g * sum(env._phi_prev) - phi0
    check(f"shaping telescopes ({steps} steps, {len(env._released)} released)",
          abs(acc - target) < 1e-6 * max(1.0, abs(target)), f"residual {abs(acc - target):.1e}")

    # 3. snapshot / restore replays exactly
    print("3. snapshot/restore")
    env = release_env(pbrs_flat_deviation=True)
    seed = int(Config.SCENARIO_EVAL_SEEDS[0])
    np.random.seed(seed % 2**32)
    obs, _ = env.reset(seed=seed, options={"scenario_seed": seed})
    for _ in range(40):
        obs, *_ = env.step(np.array([first_selectable(obs), DO_NOTHING, 0]))
    snap, obs0 = env.snapshot(), obs
    rng = np.random.default_rng(5)


    def run(obs, n=25):
        out = []
        for _ in range(n):
            a = first_selectable(obs)
            valid = np.flatnonzero(obs["mask_action_per_ac"][a])
            obs, r, te, tr, info = env.step(np.array([a, int(rng.choice(valid)), 0]))
            out.append(r)
            if te or tr:
                break
        return out, dict(env._released)


    first = run(obs0)
    env.restore(snap)
    rng = np.random.default_rng(5)
    second = run(obs0)
    check("restore replays rewards and releases exactly", first == second, f"{len(first[0])} steps, {len(first[1])} released")

    # 4. evaluation reads the run's env
    print("4. evaluation config from run_meta")
    from analysis.lookahead import env_config_for, reward_config_for

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "run_meta.json").write_text(json.dumps({"windowed_config": {
            "reward_mode": "outcome_pbrs", "release_locked": True, "release_dev_s": 30.0, "pay_at_release": True,
            "pbrs_flat_deviation": True, "landing_brackets": [list(b) for b in HARD]}}))
        model = str(run_dir / "final_model.zip")
        ecfg, rcfg = env_config_for(model), reward_config_for(model)
        check("env_config_for: the MDP keys", ecfg == {"release_locked": True, "release_dev_s": 30.0}, str(ecfg))
        check("reward_config_for: MDP + reward keys, hashable",
              rcfg["pay_at_release"] and rcfg["pbrs_flat_deviation"] and rcfg["release_locked"]
              and hash(tuple(sorted(rcfg.items()))) is not None)
        WindowedATCEnv({"evaluation_mode": True, **rcfg})  # the lookahead's env builds

    # 5. feasible-only training streams
    print("5. train_max_early_s")


    def screened(train_seed, n=4):
        env = WindowedATCEnv({"evaluation_mode": False, "reward_mode": "outcome_pbrs", "stitch_segments": 2,
                              "stitch_gap_s": (120.0, 900.0), "train_seed": train_seed, "train_max_early_s": 650.0})
        out = []
        for _ in range(n):
            env.reset()
            out.append((sorted(env._eta.values())[0], round(env._max_early_s(), 3)))
        return out, env._train_redraws


    a, redraws = screened(11)
    b, _ = screened(11)
    check("every accepted training stream is feasible (no flight > 650 s early)", all(e <= 650.0 for _, e in a),
          f"max early {[e for _, e in a]}, {redraws} redraws for {len(a)} streams")
    check("over-capacity streams were redrawn", redraws > 0)
    check("same train_seed, same accepted streams", a == b)

    # 6. streams run to their end with release on (the window is never empty, nothing crashes)
    print("6. whole streams with release, random clearances")
    worst = []
    for flights, segments, dev_s in ((10, 1, 30.0), (10, 1, 1e9), (20, 2, 30.0), (20, 2, 1e9)):
        env = WindowedATCEnv({"evaluation_mode": False, "reward_mode": "outcome_pbrs", "scenario_aircraft_count": flights,
                              "stitch_segments": segments, "stitch_gap_s": (120.0, 900.0), "train_seed": 21,
                              "landing_brackets": HARD, "release_locked": True, "release_dev_s": dev_s,
                              "pay_at_release": True, "pbrs_flat_deviation": True, "train_max_early_s": 650.0})
        rng = np.random.default_rng(flights + segments)
        ok, n_steps, n_rel, only_released = True, 0, 0, 0
        for _ in range(2):
            obs, _ = env.reset()
            done = False
            while not done and n_steps < 5000:
                sel = np.flatnonzero(obs["mask_select"])
                if len(sel) == 0 or not env._prev_aircraft_order[int(sel[0])]:
                    ok = False
                    break
                a = int(rng.choice(sel))
                valid = np.flatnonzero(obs["mask_action_per_ac"][a])
                c = DO_NOTHING if rng.random() < 0.5 else int(rng.choice(valid))
                only_released += all(cs in env._released for cs in env._prev_aircraft_order if cs)
                obs, r, te, tr, info = env.step(np.array([a, c, int(rng.random() < 0.2)]))
                done, n_steps = te or tr, n_steps + 1
            n_rel += len(env._released)
        check(f"{flights}x{segments} streams, release within {dev_s:g} s: run to the end, a selectable flight at every step",
              ok, f"{n_steps} steps, {n_rel} released, {only_released} steps showing only released flights")

    # the stream's tail, forced: every airborne flight released -> they are shown, DO_NOTHING-only, and a step runs
    env = release_env()
    seed = int(Config.SCENARIO_EVAL_SEEDS[1])
    np.random.seed(seed % 2**32)
    obs, _ = env.reset(seed=seed, options={"scenario_seed": seed})
    for _ in range(5):
        obs, *_ = env.step(np.array([first_selectable(obs), DO_NOTHING, 0]))
    for ac in env.simulator.world.aircraft:
        if not ac.mission.completed:
            env._released.setdefault(ac.call_sign, 0.0)
    obs, order = env._build_window_observation(count=False)
    env._prev_aircraft_order = order
    slots = [i for i, cs in enumerate(order) if cs]
    only_do_nothing = all(obs["mask_action_per_ac"][i].sum() == 1 and obs["mask_action_per_ac"][i][DO_NOTHING] == 1 for i in slots)
    selectable = [i for i in np.flatnonzero(obs["mask_select"]) if order[i]]
    check("all airborne released: the window shows them, DO_NOTHING only, still selectable",
          len(slots) > 0 and only_do_nothing and len(selectable) > 0, f"{len(slots)} shown, {len(selectable)} selectable")
    obs, r, te, tr, info = env.step(np.array([selectable[0], DO_NOTHING, 0]))
    check("and a step runs", True)

    print("\nALL PASS" if not failures else f"\nFAILED: {failures}")
    return failures


def test_release_contracts():
    failures = run_checks()
    assert not failures, failures


if __name__ == "__main__":
    sys.exit(0 if not run_checks() else 1)
