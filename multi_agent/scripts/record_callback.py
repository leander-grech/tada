"""Record a trained RLlib PPO model's behavior to video via an evaluation callback."""

import argparse
import sys
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.callbacks.callbacks import RLlibCallback

from common.env_utils import get_base_env
from common.runtime_log import force_runtime_log_color, runtime_print
from common.seeding import seed_all
from config.config import Config
from evaluation.loading import build_eval_algo


class EpisodeVideoRecorderCallback(RLlibCallback):
    """Callback that captures render frames and writes an episode video.

    NOTE: In multi-agent environments, RLlib may call on_episode_end for individual
    agent sub-episodes, not just when the full environment episode ends. This callback
    uses (worker_index, env_index) as keys to track videos per environment instance,
    and only finalizes videos when the environment's `last_end_reason` is set
    (indicating the full episode has ended).
    """

    def __init__(
        self,
        env_runner_indices: Sequence[int] | None = None,
        output_dir: Path | str = "videos",
        fps: int = 2,
        frame_stride: int = 1,
        max_episodes: int | None = None,
    ) -> None:
        """Initialize the video recorder callback."""
        super().__init__()
        self._env_runner_indices: Sequence[int] | None = env_runner_indices
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._frame_stride = int(max(1, frame_stride))
        self._max_episodes = max_episodes
        # Key by (worker_index, env_index) instead of episode object to handle
        # multi-agent sub-episodes correctly
        self._episode_writers: dict[tuple, Any] = {}
        self._run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._episode_counter = 0  # Track episode number to ensure unique filenames
        self._frame_counts: dict[tuple, int] = {}  # Track frame count per env_key

    # --- Helpers ---
    @staticmethod
    def _render_first_env(env: Any):
        """Render the base env (sub-env 0 if vectorized). Returns RGB ndarray."""
        base_env = get_base_env(env)
        render_fn = getattr(base_env, "render", None)
        if not callable(render_fn):
            raise AttributeError("Environment does not support render()")
        return render_fn()

    def _create_video_writer(self, env: Any, video_path: Path):
        """Create an ffmpeg writer; pre-render a frame to avoid probing.

        Returns: (writer, first_frame_or_none)
        """

        # Render the first frame up-front so the writer knows dimensions/format
        first_frame = self._render_first_env(env)

        writer = imageio.get_writer(
            video_path,
            fps=self._fps,
            ffmpeg_log_level="quiet",
        )
        return writer, first_frame

    def _should_capture(self, step: int) -> bool:
        return (step % self._frame_stride) == 0

    def _finalize_video(self, env_key: tuple, end_reason: str) -> bool:
        """Finalize and close the video for the given env_key.

        Returns True if a video was finalized, False if no video was active.
        """
        entry = self._episode_writers.pop(env_key, None)
        if entry is None:
            return False

        frame_count = self._frame_counts.pop(env_key, 0)
        writer, video_path, _ = entry
        writer.close()
        runtime_print(
            "Recorder",
            "video_saved",
            end_reason=end_reason,
            path=video_path,
            frames=frame_count,
        )
        return True

    # --- RLlib Callback API ---
    def on_episode_start(
        self,
        *,
        episode: Any,
        env_runner: Any,
        metrics_logger: Any,
        env: Any,
        env_index: int,
        rl_module: Any,
        **kwargs,
    ) -> None:
        if (
            self._env_runner_indices is not None
            and env_runner.worker_index not in self._env_runner_indices
        ):
            return

        # Use (worker_index, env_index) as key to handle multi-agent sub-episodes
        env_key = (env_runner.worker_index, env_index)
        base_env = get_base_env(env)
        seed = getattr(base_env, "scenario_seed", "unknown")

        # Check if we already have a writer for this env_key
        existing_entry = self._episode_writers.get(env_key)
        if existing_entry is not None:
            _, _, existing_seed = existing_entry
            if str(seed) != str(existing_seed):
                # Seed changed - finalize the previous episode's video
                end_reason = getattr(base_env, "last_end_reason", None) or "auto-finalized"
                self._finalize_video(env_key, end_reason)
            else:
                # Same seed - this is a sub-episode start, skip creating new writer
                return

        # Stop recording after reaching the episode limit
        if self._max_episodes is not None and self._episode_counter >= self._max_episodes:
            return

        # Increment episode counter to ensure unique filenames even if same seed repeats
        self._episode_counter += 1
        filename = f"{self._run_timestamp}_ep{self._episode_counter:02d}_seed-{seed}.mp4"
        video_path = self._output_dir / filename
        writer, first_frame = self._create_video_writer(env, video_path)
        self._episode_writers[env_key] = (writer, video_path, seed)  # Store seed for comparison
        self._frame_counts[env_key] = 1  # Count first frame
        writer.append_data(first_frame)

    def on_episode_step(
        self,
        *,
        episode: Any,
        env_runner: Any,
        metrics_logger: Any,
        env: Any,
        env_index: int,
        rl_module: Any,
        **kwargs,
    ) -> None:
        """Stream current render frame to the episode's writer."""

        env_key = (env_runner.worker_index, env_index)
        entry = self._episode_writers.get(env_key)
        if entry is None:
            return

        # Use environment's step counter instead of episode.length for consistent tracking
        base_env = get_base_env(env)
        env_step = getattr(base_env, "step_counter", 0)

        if not self._should_capture(env_step):
            return

        frame = self._render_first_env(env)
        writer, _, _ = entry
        if frame is not None:
            writer.append_data(frame)
            self._frame_counts[env_key] = self._frame_counts.get(env_key, 0) + 1

    def on_episode_end(
        self,
        *,
        episode: Any,
        env_runner: Any,
        metrics_logger: Any,
        env: Any,
        env_index: int,
        rl_module: Any,
        **kwargs,
    ) -> None:
        """Finalize video only when the FULL episode has ended.

        In multi-agent environments, RLlib may call on_episode_end for individual
        agent sub-episodes. We detect the true episode end by checking if the
        environment's `last_end_reason` is set.
        """
        env_key = (env_runner.worker_index, env_index)
        if self._episode_writers.get(env_key) is None:
            return

        base_env = get_base_env(env)
        end_reason = getattr(base_env, "last_end_reason", None)
        if end_reason is None:
            return

        self._finalize_video(env_key, end_reason)


def _make_video_recorder_callback(max_episodes: int) -> type[RLlibCallback]:
    """Factory that returns a callback class configured with an episode limit."""

    class ConfiguredVideoRecorderCallback(EpisodeVideoRecorderCallback):
        """Video recorder callback pre-configured with an episode limit."""

        def __init__(self, **kwargs):
            super().__init__(max_episodes=max_episodes, **kwargs)

    return ConfiguredVideoRecorderCallback


def load_ppo_for_recording(
    checkpoint_dir: Path | str,
    max_steps_per_episode: int,
    num_episodes: int = 1,
    max_aircraft: int = Config.MAX_AIRCRAFT_COUNT_EVAL,
    seed: int | None = None,
) -> PPO:
    """Load PPO policy weights and configure evaluation for recording."""
    callback_cls = _make_video_recorder_callback(max_episodes=num_episodes)

    # Timestep budget with 10% buffer: RLlib counts multi-agent sub-episodes
    # as full episodes, so we use timesteps to avoid premature interruption.
    timestep_budget = int(max_steps_per_episode * num_episodes * 1.1)

    return build_eval_algo(
        checkpoint_dir,
        eval_env_config={
            "max_aircraft_count": max_aircraft,
            "max_steps_per_episode": max_steps_per_episode,
            "evaluation_mode": True,
            "evaluation_randomize_scenario_count": max(
                0, max(1, num_episodes) - len(Config.SCENARIO_EVAL_SEEDS)
            ),
        },
        callbacks=[callback_cls],
        timestep_budget=timestep_budget,
        seed=seed,
    )


def main() -> None:
    force_runtime_log_color()
    parser = argparse.ArgumentParser(
        description=(
            "Record PPO policy behavior to video using evaluation with a recording callback."
        )
    )
    parser.add_argument(
        "-m",
        "--max-steps",
        dest="max_steps",
        type=int,
        required=False,
        default=Config.EPISODE_STEPS,
        help="Maximum steps per episode for recording.",
    )
    parser.add_argument(
        "-d",
        "--model-directory",
        dest="model_directory",
        type=str,
        required=True,
        help="Directory containing the RLlib PPO checkpoint.",
    )
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        required=False,
        default=1,
        help="Number of evaluation episodes to record.",
    )
    parser.add_argument(
        "-a",
        "--max-aircraft",
        type=int,
        default=Config.MAX_AIRCRAFT_COUNT_EVAL,
        help=f"Max aircraft count (default: {Config.MAX_AIRCRAFT_COUNT_EVAL}).",
    )
    args = parser.parse_args()

    seed = Config.SEED
    seed_all(seed, deterministic=True)

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray\.")
    ray.init(ignore_reinit_error=True)

    model_dir: Path = Path(args.model_directory).absolute()
    if not model_dir.is_dir():
        runtime_print(
            "Recorder",
            "error",
            message=f"no such model directory {args.model_directory}",
            file=sys.stderr,
        )
        sys.exit(1)

    algo = load_ppo_for_recording(
        model_dir, args.max_steps, args.num_episodes, args.max_aircraft, seed=seed
    )
    _ = algo.evaluate()
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
