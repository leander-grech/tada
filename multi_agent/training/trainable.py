"""One fixed Algorithm per Tune segment with complete owned checkpoint bundles."""

import time
from pathlib import Path

from ray.rllib.algorithms.ppo import PPO
from ray.tune import Trainable

from training.checkpoint_boundary import capture_boundary_state
from training.checkpoints import capture_checkpoint, inspect_checkpoint, load_checkpoint_algorithm
from training.ppo_config import build_algorithm, build_ppo_config
from training.provenance import write_run_manifest
from training.step_hooks import StepHooks


class SingleRunTrainable(Trainable):
    """Keep Tune segment counters separate from Algorithm lifetime counters."""

    @classmethod
    def default_resource_request(cls, config):
        return PPO.default_resource_request(build_ppo_config(config["run"]).to_dict())

    def setup(self, config):
        self.algorithm = None
        self._last_step_completed = False
        self._checkpoint_saved_iteration = None
        self.run = config["run"]
        self.segment_dir = Path(config["segment_dir"])
        checkpoint = config.get("checkpoint")
        if checkpoint is None:
            self.algorithm = build_algorithm(
                self.run, scenario_artifact_dir=self.segment_dir / "scenarios"
            )
        else:
            loaded = inspect_checkpoint(
                Path(checkpoint),
                self.run,
                mode=config.get("mode", "resume"),
                changes=config.get("changes", {}),
            )
            self.algorithm = load_checkpoint_algorithm(
                loaded,
                self.run,
                scenario_artifact_dir=self.segment_dir / "scenarios",
            )
        try:
            setattr(self.algorithm, "_checkpoint_execution_segment", config["segment"])
            self.hooks = StepHooks(
                self.run, self.segment_dir, self.algorithm, env_steps=None if checkpoint else 0
            )
            write_run_manifest(
                self.segment_dir / "manifest.json",
                self.run,
                {},
                algorithm=self.algorithm,
                execution_segment=config["segment"],
            )
        except BaseException:
            self.cleanup()
            raise

    def step(self):
        assert self.algorithm is not None
        self._last_step_completed = False
        self.hooks.before_step()
        started = time.monotonic()
        previous_evaluation = getattr(self.algorithm, "_operational_evaluation_seconds", 0.0)
        result = self.algorithm.train()
        elapsed = time.monotonic() - started
        evaluation = (
            getattr(self.algorithm, "_operational_evaluation_seconds", 0.0) - previous_evaluation
        )
        # This counter is produced by RLlib's actual sampling, not Tune's iteration.
        steps = result.get("num_env_steps_sampled_lifetime")
        if steps is None:
            steps = result.get("env_runners", {}).get("num_env_steps_sampled_lifetime")
        if steps is None:
            raise RuntimeError("Algorithm omitted lifetime environment-step evidence")
        result["lifetime_env_steps"] = int(steps)
        result["lifetime_iterations"] = self.algorithm.training_iteration
        result.pop("training_iteration", None)
        result["phase_iterations"] = self.algorithm.training_iteration - getattr(
            self.algorithm, "_checkpoint_phase_start", 0
        )
        self._last_step_completed = True
        return self.hooks.after_step(result, elapsed_seconds=elapsed, evaluation_seconds=evaluation)

    def save_checkpoint(self, checkpoint_dir):
        if getattr(self, "algorithm", None) is None:
            raise ValueError("unsupported checkpoint contract: no owned Algorithm")
        completed = capture_checkpoint(
            self.algorithm, Path(checkpoint_dir), self.run, advertise_latest=False
        )
        self._checkpoint_saved_iteration = getattr(self.algorithm, "training_iteration", None)
        return str(completed)

    def load_checkpoint(self, checkpoint):
        if not hasattr(self, "run"):
            raise ValueError("unsupported checkpoint contract: no resolved run")
        if (
            getattr(self, "_iteration", 0) != 0
            or getattr(self, "_last_result", None)
            or (self.algorithm is not None and self.algorithm.training_iteration != 0)
        ):
            raise ValueError("checkpoint load requires a fresh unreported Tune segment")
        loaded = inspect_checkpoint(Path(checkpoint), self.run)
        if self.algorithm is not None:
            # Do not discard independently sampled/unlearned work, even before
            # this Trainable's first report. Reuse the owned quiescence proof.
            capture_boundary_state(self.algorithm, self.run)
        previous, self.algorithm = self.algorithm, None
        if previous is not None:
            previous.stop()
        self._last_step_completed = False
        self.algorithm = load_checkpoint_algorithm(
            loaded, self.run, scenario_artifact_dir=self.segment_dir / "scenarios"
        )
        self._checkpoint_saved_iteration = getattr(self.algorithm, "training_iteration", None)
        segment = getattr(self, "config", {}).get("segment")
        if segment is not None:
            setattr(self.algorithm, "_checkpoint_execution_segment", segment)
        self.hooks = StepHooks(self.run, self.segment_dir, self.algorithm, env_steps=None)

    def _report_class_trainable_checkpoint(self, checkpoint_dir, checkpoint_dict_or_path):
        from training.checkpoint_tune import report_completed_checkpoint

        return report_completed_checkpoint(self, checkpoint_dir, checkpoint_dict_or_path)

    def reset_config(self, new_config):
        raise ValueError("unsupported mutable configuration contract: start a fixed execution")

    def cleanup(self):
        if self.algorithm is not None:
            try:
                if (
                    getattr(self, "_last_step_completed", False)
                    and self._checkpoint_saved_iteration != self.algorithm.training_iteration
                ):
                    capture_checkpoint(
                        self.algorithm,
                        self.segment_dir
                        / "stop"
                        / f"checkpoint_{self.algorithm.training_iteration:06d}",
                        self.run,
                    )
            finally:
                self.algorithm.stop()
                self.algorithm = None
