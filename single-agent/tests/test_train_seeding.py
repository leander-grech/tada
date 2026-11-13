"""Reproducible training scenarios (env config "train_seed") and the flat potentials.

  1. Two training envs with the same train_seed draw the same scenarios, frames and rewards for
     the same actions; a different train_seed draws different scenarios; no draw is one of the
     validation seeds (which training must never see).
  2. "pbrs_flat_potentials": on the same state, the sequence and conflict potentials are at least
     as large in magnitude as the ramped ones (every weight is 1 instead of <= 1), the deviation
     and gap potentials are unchanged, and the shaping still telescopes over an episode.

Run: TADA_SEQUENCE_OBS=1 TADA_MAX_PICKS=2 OMP_NUM_THREADS=1 python tests/test_train_seeding.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from atc_env.windowed_env import WindowedATCEnv
from config.config import Config

GAMMA = float(Config.PBRS_GAMMA)
def run_checks():
    failures = []


    def check(name, ok, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)


    def make(train_seed, flat=False):
        return WindowedATCEnv({"evaluation_mode": False, "reward_mode": "outcome_pbrs", "stitch_segments": 2,
                               "stitch_gap_s": (120.0, 900.0), "train_seed": train_seed, "pbrs_flat_potentials": flat})


    def play(env, n_resets=2, n_steps=12, action_seed=0):
        """Scenario seeds, first observations and per-step rewards for a fixed action stream."""
        rng = np.random.default_rng(action_seed)
        out = []
        for _ in range(n_resets):
            obs, _ = env.reset()
            rec = {"seed": env.scenario_seed, "obs0": {k: v.copy() for k, v in obs.items()}, "r": [],
                   "gaps": list(env.simulator.stitch_gaps_last)}
            for _ in range(n_steps):
                sel = np.flatnonzero(obs["mask_select"]) if "mask_select" in obs else np.flatnonzero(obs["mask_aircraft"])
                a = int(rng.choice(sel))
                valid = np.flatnonzero(obs["mask_action_per_ac"][a])
                c = 0 if rng.random() < 0.5 else int(rng.choice(valid))
                act = [a, c, 0] if "mask_select" in obs else [a, c]
                obs, r, te, tr, _ = env.step(np.array(act))
                rec["r"].append(float(r))
                if te or tr:
                    break
            out.append(rec)
        return out


    print("1. seeded training scenarios")
    a1, a2, b = play(make(7)), play(make(7)), play(make(8))
    same = all(x["seed"] == y["seed"] and x["gaps"] == y["gaps"] and x["r"] == y["r"]
               and all(np.array_equal(x["obs0"][k], y["obs0"][k]) for k in x["obs0"]) for x, y in zip(a1, a2))
    check("same train_seed: same scenarios, gaps, first observations and rewards", same,
          f"seeds {[x['seed'] for x in a1]}")
    check("different train_seed: different scenarios", [x["seed"] for x in a1] != [x["seed"] for x in b],
          f"seeds {[x['seed'] for x in b]}")
    holdout = {int(s) for s in Config.SCENARIO_EVAL_SEEDS}
    check("no drawn scenario is a validation seed", not any(x["seed"] in holdout for x in a1 + b),
          f"{len(holdout)} validation seeds held out")
    # Force a collision: make the generator's first draw a "validation seed"; reset must skip it.
    peek = np.random.default_rng(11)
    first, second = int(peek.integers(0, 2**31 - 1)), int(peek.integers(0, 2**31 - 1))
    env = make(11)
    env._holdout_seeds = env._holdout_seeds | {first}
    env.reset()
    check("a draw that is a validation seed is skipped", env.scenario_seed == second and env.scenario_seed != first,
          f"first draw {first} held out -> scenario {env.scenario_seed}")

    print("2. flat potentials")
    ramp, flat = make(21), make(21, flat=True)
    ramp.reset(); flat.reset()
    pr, pf = ramp._potential(), flat._potential()
    check("same scenario for both", ramp.scenario_seed == flat.scenario_seed)
    check("deviation and gap potentials unchanged", abs(pr[0] - pf[0]) < 1e-9 and abs(pr[3] - pf[3]) < 1e-9,
          f"dev {pr[0]:.3f}/{pf[0]:.3f}, gap {pr[3]:.3f}/{pf[3]:.3f}")
    from actions.rewards import infringement_reward_from_world
    from config.windowed_config import WindowedConfig
    now = float(flat.simulator.world.age_in_seconds)
    expect = infringement_reward_from_world(flat.current_predicted_world, now, weight=WindowedConfig.PHI_CONFLICT_WEIGHT_FLAT,
                                            callsigns=flat._window_callsigns(), flat=True)
    check("sequence potential: every inversion counts in full (magnitude >= ramped)", abs(pf[2]) >= abs(pr[2]) - 1e-9,
          f"seq {pr[2]:.3f} -> {pf[2]:.3f}")
    check("conflict potential: undiscounted severity sum at the flat weight", abs(pf[1] - expect) < 1e-9,
          f"conflict {pr[1]:.3f} (ramped, weight {WindowedConfig.PHI_CONFLICT_WEIGHT:.0f}) -> {pf[1]:.3f} (flat, weight {WindowedConfig.PHI_CONFLICT_WEIGHT_FLAT:.0f})")
    # telescoping over an episode with the flat potentials (one pick per step here: discount gamma)
    env = make(33, flat=True)
    obs, _ = env.reset()
    phi0 = sum(env._potential())
    rng = np.random.default_rng(1)
    acc, g, done, steps = 0.0, 1.0, False, 0
    while not done and steps < 60:
        a = int(rng.choice(np.flatnonzero(obs["mask_select"])))
        valid = np.flatnonzero(obs["mask_action_per_ac"][a])
        c = 0 if rng.random() < 0.6 else int(rng.choice(valid))
        obs, r, te, tr, info = env.step(np.array([a, c, 0]))
        p = info["reward_parts"]
        acc += g * (p.deviation + p.conflict + p.goal)
        g *= 1.0 if info.get("zero_time") else GAMMA
        done = te or tr
        steps += 1
    target = g * sum(env._phi_prev) - phi0
    check(f"shaping telescopes with flat potentials ({steps} steps)", abs(acc - target) < 1e-6 * max(1.0, abs(target)),
          f"residual {abs(acc - target):.1e}")

    print("\nALL PASS" if not failures else f"\nFAILED: {failures}")
    return failures


def test_training_seed_contracts():
    failures = run_checks()
    assert not failures, failures


if __name__ == "__main__":
    sys.exit(0 if not run_checks() else 1)
