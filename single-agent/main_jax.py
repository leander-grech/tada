"""Train the ATC policy with PPO in JAX (jax_ppo/), writing SB3-compatible checkpoints.

Same environments, reward normalisation, schedules, critic warm-up, run directories and
TensorBoard metrics compatible with SB3. What
changes is the learner: jax_ppo/ppo.py reproduces SB3's PPO update to float precision
(tests/test_jax_ppo_update.py), jitted on the GPU.

Checkpoints are ordinary SB3 PPO .zip files (+ VecNormalize .pkl) with SB3's names, so
render_policy.py, analysis/score_windowed.py, analysis/lookahead.py and the rest load them unchanged.

    TADA_SEQUENCE_OBS=1 python -u main_jax.py --env windowed --reward-mode outcome_pbrs \\
        --init-weights experiments/<run>/checkpoints/ppo_tada_<N>_steps.zip \\
        --n-envs 16 --critic-warmup-steps 100000 --lr-max 3e-5 --final-lr 3e-6 \\
        --ent-coef 0.003 --total-timesteps 5000000 --run-suffix windowed_jax

Reselection (up to N clearances per 45 s step, the policy asking for each extra pick):
TADA_MAX_PICKS=N in the environment, as TADA_SEQUENCE_OBS (it sizes the spaces). Requires
--reward-mode outcome_pbrs. A pre-reselection --init-weights checkpoint is extended with a fresh
"again" head and a zero-initialised "cleared at this step" column.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # avoid reserving all device memory up front

import argparse
import copy
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import main as sb3_main  # run naming, metadata and schedules
from actions.action_set import load_policy_at_own_width
from actions.actions import Action
from atc_env.single_agent_env import ATCSingleAgentEnv
from atc_env.windowed_env import WindowedATCEnv
from callbacks.metrics_callback import ATCEpisodeMetricsCallback
from config.config import Config
from config.windowed_config import WindowedConfig
from jax_ppo.export import params_from_policy, save_sb3
from jax_ppo.ppo import JaxPPO, PPOConfig
from models.observations import AircraftScalarFeatures
from network.autoregressive_policy import ATCAutoregressivePolicy, warm_start_policy


class _CallbackModelShim:
    """The slice of an SB3 model that ATCEpisodeMetricsCallback reads (logger, env, timesteps)."""

    def __init__(self, logger, venv):
        self.logger = logger
        self._venv = venv
        self.num_timesteps = 0

    def get_env(self):
        return self._venv


def parse_args():
    ap = argparse.ArgumentParser(description="ATC PPO trainer, JAX learner, SB3-compatible output")
    ap.add_argument("--env", choices=["default", "windowed"], default="windowed")
    ap.add_argument("--reward-mode", choices=["legacy", "outcome_pbrs"], default=None)
    ap.add_argument("--stitch-segments", type=int, default=None)
    ap.add_argument("--stitch-gap", type=float, nargs=2, default=None)
    ap.add_argument("--flights", type=int, default=None, help="windowed: flights per scenario segment (default 20)")
    ap.add_argument("--curriculum", default=None,
                    help='windowed: traffic stages "flights:segments:until_step[:pool],..." in one run, e.g. '
                         '"10:1:2500000,20:1:5500000,20:2:10000000". Policy, optimiser, reward normalisation '
                         'and the LR schedule carry over; only the env workers are replaced at each boundary. '
                         'The optional pool is a seed file for that stage (overrides --train-seed-pool), for a '
                         'difficulty curriculum: analysis/curriculum_pools.py')
    ap.add_argument("--pbrs-flat", action="store_true",
                    help="outcome_pbrs: sequence and conflict potentials without their time ramps")
    ap.add_argument("--violation-penalty", type=float, default=None,
                    help="outcome_pbrs bust penalty (default WindowedConfig.PBRS_VIOLATION_PENALTY = 90)")
    ap.add_argument("--hard-bracket", action="store_true",
                    help="outcome_pbrs: target +/-30 s (WindowedConfig.HARD_BRACKETS: a top bracket worth 4.0; "
                         "the bust penalty follows, 90 -> 100)")
    ap.add_argument("--release-locked", action="store_true",
                    help="windowed: hand off the front of the landing queue while each flight is locked "
                         "(predicted within --release-dev-s, no predicted conflict); it leaves the window")
    ap.add_argument("--release-dev-s", type=float, default=30.0)
    ap.add_argument("--pay-at-release", action="store_true",
                    help="outcome_pbrs + --release-locked: pay a flight's bracket at release, correct at touchdown")
    ap.add_argument("--flat-deviation", action="store_true",
                    help="outcome_pbrs: deviation potential at full weight for every flight the prediction resolves")
    ap.add_argument("--train-max-early-s", type=float, default=None,
                    help="windowed: train on feasible streams only -- redraw a training scenario while a flight "
                         "is predicted more than this early (650 = the generator's cap; the evaluation's filter)")
    ap.add_argument("--train-seed-pool", default=None,
                    help="windowed: draw training scenarios from this file of pre-screened seeds (one per line), "
                         "e.g. feasible streams from analysis/long_stream_seeds.py; needs --seed")
    ap.add_argument("--use-case", type=int, choices=(1, 2), default=None,
                    help="windowed: scenario, 1 = MXP trombone (default), 2 = BGY point merge")
    ap.add_argument("--init-weights", default=None, help="SB3 .zip to start from (widened if needed)")
    ap.add_argument("--total-timesteps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--lr-max", type=float, default=3e-4)
    ap.add_argument("--final-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.08)
    ap.add_argument("--critic-warmup-steps", type=int, default=0)
    ap.add_argument("--critic-warmup-lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.25)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--save-freq", type=int, default=25_000, help="checkpoint cadence, timesteps")
    ap.add_argument("--run-suffix", default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    n_envs = int(args.n_envs)
    if sb3_main.ROLLOUT_BUFFER_SIZE % n_envs:
        raise SystemExit(f"--n-envs must divide {sb3_main.ROLLOUT_BUFFER_SIZE}")
    n_steps = sb3_main.ROLLOUT_BUFFER_SIZE // n_envs

    run_name = sb3_main._next_run_name("atc_run_1")
    if args.run_suffix:
        run_name = f"{run_name}_{args.run_suffix.strip().strip('_')}"
    exp_dir = Path("experiments") / run_name
    ckpt_dir, tb_dir = exp_dir / "checkpoints", exp_dir / "tb"
    for d in (ckpt_dir, tb_dir, exp_dir / "best", exp_dir / "eval_logs"):
        d.mkdir(parents=True, exist_ok=True)
    sb3_main._write_initial_run_meta(exp_dir, run_name, args.total_timesteps)
    print(f"[experiment] {run_name} -> {exp_dir}  (JAX learner)")

    env_cls = WindowedATCEnv if args.env == "windowed" else ATCSingleAgentEnv
    max_picks = int(Config.MAX_PICKS)
    if max_picks > 1:
        mode = args.reward_mode or WindowedConfig.REWARD_MODE
        if args.env != "windowed" or mode != "outcome_pbrs":
            raise SystemExit("TADA_MAX_PICKS > 1 needs --env windowed --reward-mode outcome_pbrs: the "
                             "legacy reward's per-step dense terms would be charged again at every pick")
    wcfg = {}
    if args.stitch_segments is not None:
        wcfg["stitch_segments"] = int(args.stitch_segments)
    if args.stitch_gap is not None:
        wcfg["stitch_gap_s"] = tuple(args.stitch_gap)
    if args.reward_mode is not None:
        wcfg["reward_mode"] = args.reward_mode
    if args.violation_penalty is not None:
        wcfg["pbrs_violation_penalty"] = float(args.violation_penalty)
    if args.pbrs_flat:
        wcfg["pbrs_flat_potentials"] = True
    if args.hard_bracket:
        wcfg["landing_brackets"] = [list(b) for b in WindowedConfig.HARD_BRACKETS]
    if args.release_locked:
        wcfg["release_locked"] = True
        wcfg["release_dev_s"] = float(args.release_dev_s)
    if args.pay_at_release:
        wcfg["pay_at_release"] = True
    if args.flat_deviation:
        wcfg["pbrs_flat_deviation"] = True
    if args.train_max_early_s is not None:
        wcfg["train_max_early_s"] = float(args.train_max_early_s)
    if args.train_seed_pool:
        wcfg["train_seed_pool"] = str(args.train_seed_pool)
    if args.use_case is not None:
        wcfg["use_case"] = int(args.use_case)
    if args.flights is not None:
        wcfg["scenario_aircraft_count"] = int(args.flights)
    stages = []
    if args.curriculum:
        if env_cls is not WindowedATCEnv:
            raise SystemExit("--curriculum needs --env windowed")
        for part in args.curriculum.split(","):
            fields = part.split(":")
            f, seg, until = (int(float(x)) for x in fields[:3])
            stage = {"flights": f, "segments": seg, "until": until}
            if len(fields) > 3 and fields[3]:
                # Optional 4th field: this stage's training-scenario pool (a seed file, as
                # --train-seed-pool), e.g. difficulty tiers easy -> easy+medium -> all feasible.
                if not Path(fields[3]).is_file():
                    raise SystemExit(f"--curriculum: stage pool {fields[3]!r} not found")
                stage["pool"] = fields[3]
            stages.append(stage)
        if any(b["until"] <= a["until"] for a, b in zip(stages, stages[1:])) or stages[-1]["until"] < args.total_timesteps:
            raise SystemExit("--curriculum: 'until' steps must increase and the last must reach --total-timesteps")
        assert n_envs < 100, "per-stage worker seeds assume fewer than 100 workers"

    # Reproducible training: worker i draws its scenarios from a generator seeded with
    # seed * 1000 + i (windowed env; + 100 * stage under a curriculum), so a run is determined by
    # --seed, --n-envs and the curriculum.
    def stage_cfg(k):
        if not stages:
            return {}
        st = stages[k]
        return {"scenario_aircraft_count": st["flights"], "stitch_segments": st["segments"],
                "max_episode_steps_cap": int(WindowedConfig.MAX_EPISODE_STEPS) * st["segments"],
                **({"train_seed_pool": st["pool"]} if st.get("pool") else {})}

    def make_env(i, k=0):
        seeded = {"train_seed": int(args.seed) * 1000 + 100 * k + i} if env_cls is WindowedATCEnv else {}
        cfg = {"evaluation_mode": False, **wcfg, **stage_cfg(k), **seeded}
        return lambda: Monitor(env_cls(config=cfg))

    def build_venv(k=0):
        return VecNormalize(SubprocVecEnv([make_env(i, k) for i in range(n_envs)]),
                            norm_obs=False, norm_reward=True, gamma=sb3_main.GAMMA)

    stage = 0
    venv = build_venv(stage)
    if stages:
        print(f"[curriculum] stage 0: {stages[0]['flights']} flights x {stages[0]['segments']} segment(s) until {stages[0]['until']:,}" + (f" pool {stages[0]['pool']}" if stages[0].get('pool') else ""))

    base_lr = sb3_main.warmup_cosine_schedule(lr_max=args.lr_max, lr_min=args.final_lr, warmup_frac=args.warmup_frac)
    lr_schedule = (sb3_main.critic_warmup_schedule(base_lr, total_timesteps=args.total_timesteps,
                                                   warmup_steps=args.critic_warmup_steps,
                                                   critic_lr=args.critic_warmup_lr)
                   if args.critic_warmup_steps > 0 else base_lr)

    # SB3 template: builds the policy used by the learner (its init is the JAX init for a
    # fresh run) and is the vessel every checkpoint is saved through.
    template = PPO(ATCAutoregressivePolicy, venv, n_steps=n_steps, batch_size=256, n_epochs=5,
                   gamma=sb3_main.GAMMA, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
                   vf_coef=args.vf_coef, max_grad_norm=1.5, target_kl=args.target_kl,
                   learning_rate=lr_schedule, device="cpu", seed=args.seed)
    if args.init_weights:
        src = load_policy_at_own_width(PPO, args.init_weights, device="cpu")
        widened = warm_start_policy(template.policy, src.policy)
        print(f"[init-weights] {args.init_weights}" + (f" (widened: {widened})" if widened else ""))
    params = params_from_policy(template.policy)

    cfg = PPOConfig(n_steps=n_steps, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                    target_kl=args.target_kl, gamma=sb3_main.GAMMA,
                    heads=int(Config.ATTENTION_HEADS),
                    under_control_col=list(AircraftScalarFeatures.__dataclass_fields__).index("under_control"),
                    critic_warmup_steps=int(args.critic_warmup_steps),
                    do_nothing=int(Action.DO_NOTHING))
    agent = JaxPPO(params, venv, cfg, lr_schedule, seed=args.seed)
    if args.critic_warmup_steps > 0:
        agent.set_frozen(True)
        print(f"[critic-warmup] t=0: policy frozen except the value head")

    sb3_main._update_run_meta(exp_dir, trainer="jax", env=args.env, n_envs=n_envs,
                              windowed_config=wcfg or None, init_weights=args.init_weights,
                              sequence_obs=bool(Config.SEQUENCE_OBS),
                              aircraft_scalar_len=int(Config.AIRCRAFT_SCALAR_LEN),
                              max_picks=max_picks, seed=int(args.seed), curriculum=stages or None,
                              action_set=str(Config.ACTION_SET),
                              airport_mask=Config.AIRPORT_MASK,
                              mask_frame_fix=bool(Config.MASK_FRAME_FIX),
                              turn_mask_fix=bool(Config.TURN_MASK_FIX),
                              train_seeds=("seed * 1000 + worker index" if env_cls is WindowedATCEnv else None),
                              ent_coef=args.ent_coef, vf_coef=args.vf_coef, target_kl=args.target_kl,
                              lr_schedule={"type": "warmup_cosine", "lr_max": args.lr_max,
                                           "lr_min": args.final_lr, "warmup_frac": args.warmup_frac},
                              critic_warmup={"steps": args.critic_warmup_steps, "lr": args.critic_warmup_lr}
                              if args.critic_warmup_steps > 0 else None)

    logger = configure(str(tb_dir), ["stdout", "tensorboard", "csv"])
    for fmt in logger.output_formats:
        if isinstance(fmt, HumanOutputFormat):
            fmt.max_length = 120
    shim = _CallbackModelShim(logger, venv)
    metrics = ATCEpisodeMetricsCallback(rolling_window=100)
    metrics.init_callback(shim)
    metrics.on_training_start({}, {})

    ep_info = deque(maxlen=100)
    parts_acc = {}

    def on_step(loc):
        shim.num_timesteps = agent.num_timesteps
        metrics.update_locals(loc)
        metrics.on_step()
        for info in loc["infos"]:
            if "episode" in info:
                ep_info.append(info["episode"])
            rp = info.get("reward_parts")
            if rp is not None:
                for k in ("deviation", "conflict", "action", "bonus", "goal"):
                    parts_acc.setdefault(k, []).append(float(getattr(rp, k)))
            if "clearance" in info:
                parts_acc.setdefault("do_nothing", []).append(float(int(info["clearance"]) == 0))
            if max_picks > 1:
                parts_acc.setdefault("again", []).append(float(info.get("zero_time", False)))

    obs = venv.reset()
    episode_starts = np.ones(n_envs, dtype=np.float32)
    start, next_save, iteration = time.time(), int(args.save_freq), 0
    try:
        while agent.num_timesteps < args.total_timesteps:
            t0 = time.time()
            rollout, obs, episode_starts, t_env = agent.collect(obs, episode_starts, on_step)
            t_collect = time.time() - t0
            progress = 1.0 - agent.num_timesteps / float(args.total_timesteps)
            if agent.frozen and agent.num_timesteps >= args.critic_warmup_steps:
                agent.set_frozen(False)
                print(f"[critic-warmup] t={agent.num_timesteps:,}: policy unfrozen (fresh Adam state)")
            t1 = time.time()
            stats = agent.train(rollout, progress)
            t_train = time.time() - t1
            iteration += 1

            for k, v in stats.items():
                logger.record(k, v)
            if ep_info:
                logger.record("rollout/ep_rew_mean", float(np.mean([e["r"] for e in ep_info])))
                logger.record("rollout/ep_len_mean", float(np.mean([e["l"] for e in ep_info])))
            for k, v in parts_acc.items():
                name = {"do_nothing": "instr/action/clearance_00_frac",
                        "again": "instr/action/again_frac"}.get(k, f"instr/reward/{k}_mean")
                logger.record(name, float(np.mean(v)))
            parts_acc.clear()
            elapsed = time.time() - start
            logger.record("time/iterations", iteration)
            logger.record("time/total_timesteps", agent.num_timesteps)
            logger.record("time/time_elapsed", int(elapsed))
            logger.record("time/fps", int(agent.num_timesteps / max(elapsed, 1e-9)))
            logger.record("time/collect_s", t_collect)
            logger.record("time/env_wait_s", t_env)
            logger.record("time/train_s", t_train)
            if stages:
                logger.record("time/curriculum_stage", stage)
            logger.dump(step=agent.num_timesteps)

            if agent.num_timesteps >= next_save:
                path = ckpt_dir / f"ppo_tada_{agent.num_timesteps}_steps"
                save_sb3(template, agent.params, agent.num_timesteps, str(path))
                venv.save(str(ckpt_dir / f"ppo_tada_vecnormalize_{agent.num_timesteps}_steps.pkl"))
                while next_save <= agent.num_timesteps:
                    next_save += int(args.save_freq)

            if stages and stage + 1 < len(stages) and agent.num_timesteps >= stages[stage]["until"]:
                # Next traffic stage: new env workers; the reward-normalisation statistics carry over.
                stage += 1
                # Close the old workers BEFORE starting the new ones: building first briefly doubles the
                # process/thread count and can exceed the process limit.
                ret_rms = copy.deepcopy(venv.ret_rms)
                venv.close()
                new = build_venv(stage)
                new.ret_rms = ret_rms
                venv = agent.venv = shim._venv = new
                obs = venv.reset()
                episode_starts = np.ones(n_envs, dtype=np.float32)
                st = stages[stage]
                print(f"[curriculum] t={agent.num_timesteps:,}: stage {stage}: {st['flights']} flights x "
                      f"{st['segments']} segment(s) until {st['until']:,}" + (f" pool {st['pool']}" if st.get("pool") else ""))
    except KeyboardInterrupt:
        print("[interrupt] saving latest_model")
        save_sb3(template, agent.params, agent.num_timesteps, str(exp_dir / "latest_model"))
        venv.save(str(exp_dir / "latest_vecnormalize.pkl"))
        raise
    save_sb3(template, agent.params, agent.num_timesteps, str(exp_dir / "final_model"))
    venv.save(str(exp_dir / "vecnormalize.pkl"))
    print(f"[experiment] saved final model + vecnormalize to {exp_dir}")
    venv.close()


if __name__ == "__main__":
    main()
