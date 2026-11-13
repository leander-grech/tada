"""Training-contract environment: the training layer owns ATCEnv's run binding.

``ATCEnv`` stays a standalone simulator environment. This subclass binds it to a
verified reserved-manifest ``resolved_env`` contract, the scenario journal and
checkpoint boundary state.
"""

import json
import random
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from flight_simulator import MissionDifficulty

from actions.action_mask import detect_airport_from_route
from atc_env.environment import ATCEnv
from common.hashing import FrozenDict, content_digest
from common.runtime_log import runtime_print
from common.sealed import consume
from outcomes import vocabulary
from training.checkpoint_envelopes import validate_environment_envelope
from training.frozen_settings import (
    read_current_initial_preprocessing_settings,
    read_current_physics_settings,
    require_initial_preprocessing_window,
)
from training.generator_variant import scenario_config_root
from training.provenance import append_scenario_event
from training.reservations import effective_reserved_seeds
from training.run_config import validate_reserved_conditions
from training.runtime_gate import verify_native_artifact
from training.seed_stream import ScenarioSeedStream

# Recent failures kept per environment; older ones are dropped rather than accumulated.
FAILED_SEED_CAPACITY = 32


class TrainingATCEnv(ATCEnv):
    """``ATCEnv`` bound to an immutable reserved-manifest training contract."""

    def _verify_inference_checkpoint(self, checkpoint, payload) -> None:
        raise ValueError("inference checkpoints require the inference environment reader")

    def _configure_training_contract(self, config: dict) -> None:
        payload = config.get("resolved_env")
        if not isinstance(payload, dict) or not payload.get("reserved_manifest"):
            raise ValueError(
                "training requires an explicit reserved-manifest resolved_env contract"
            )
        inference_checkpoint = config.get("inference_checkpoint")
        if inference_checkpoint is not None:
            self._verify_inference_checkpoint(inference_checkpoint, payload)
        self._inference_checkpoint = inference_checkpoint
        allowed = {
            "resolved_env",
            "resolved_metrics",
            "expected_semantic_digest",
            "scenario_artifact_dir",
            "training_stream_state",
            "worker_index",
            "vector_index",
            "evaluation_mode",
            "require_runtime_activation",
        }
        if inference_checkpoint is not None:
            allowed |= {
                "inference_checkpoint",
                "inference_topology",
                "evaluation_seeds",
                "evaluation_seed_shards",
                "evaluation_randomize_scenario_count",
                "policy_time_limit_steps",
            }
        if set(config) - allowed:
            raise ValueError("training options must be transported in the resolved_env payload")
        payload = FrozenDict(payload)
        expected = config.get("expected_semantic_digest")
        if not isinstance(expected, str):
            raise ValueError("expected_semantic_digest is required for training")
        self.consumed_semantic_digest = consume(payload, expected, role="env")
        require_initial_preprocessing_window(payload["thresholds"], payload["reserved_conditions"])
        reserved = effective_reserved_seeds(payload["reserved_manifest"])
        validate_reserved_conditions(payload["reserved_conditions"], payload["reserved_manifest"])
        verify_native_artifact(payload["reserved_conditions"])
        if payload["reserved_conditions"]["physics"] != read_current_physics_settings():
            raise ValueError("current physics differs from frozen reservation conditions")
        if payload["reserved_conditions"]["version"] == 2:
            actual = read_current_initial_preprocessing_settings()
            if any(
                payload["reserved_conditions"]["benchmark"][key] != value
                for key, value in actual.items()
            ):
                raise ValueError(
                    "current initial preprocessing differs from frozen reservation conditions"
                )
        topology = payload["topology"]
        reader_topology = config.get("inference_topology", topology)
        if inference_checkpoint is not None and (
            set(reader_topology) != {"env_runners", "envs_per_runner"}
            or type(reader_topology["env_runners"]) is not int
            or reader_topology["env_runners"] < 0
            or reader_topology["envs_per_runner"] != 1
        ):
            raise ValueError("invalid declared inference topology")
        if (
            self._worker_index > reader_topology["env_runners"]
            or self._vector_index >= reader_topology["envs_per_runner"]
        ):
            raise ValueError("environment index exceeds resolved runner topology")
        self._training_stream = ScenarioSeedStream(
            payload["root_seed"],
            (topology["phase_slot"], self._worker_index, self._vector_index),
            reserved,
            domain="operational_evaluation" if self.evaluation_mode else "training",
        )
        directory = config.get("scenario_artifact_dir")
        if not directory:
            raise ValueError(
                "training requires scenario_artifact_dir for actual scenario identities"
            )
        identity = (
            self._training_stream.domain
            + "-"
            + "-".join(str(item) for item in self._training_stream.stream_id)
        )
        self._scenario_artifact_path = Path(directory) / f"scenarios-{identity}.jsonl"
        scenario = payload["scenario"]
        self._scenario_config_root = scenario_config_root(scenario)
        difficulty = getattr(MissionDifficulty, scenario["difficulty"], None)
        if difficulty is None:
            raise ValueError("scenario difficulty is unsupported by the installed native artifact")
        self.difficulty = difficulty
        if self.settings.env.world_json_file is not None:
            raise ValueError("reserved training cannot use WORLD_JSON_FILE")
        self._airport = scenario["airport"]
        if mask := scenario.get("custom_action_mask"):
            action = self.settings.core.action
            airports = {
                **action.custom_action_mask_by_airport,
                self._airport: tuple(map(int, mask)),
            }
            self.settings = replace(
                self.settings,
                core=replace(
                    self.settings.core,
                    action=replace(
                        action, custom_action_mask_by_airport=MappingProxyType(airports)
                    ),
                ),
            )
        self.max_aircraft_count = scenario["aircraft_count"]
        # Checkpoints saved before the field existed train on a fixed count.
        self._min_aircraft_count = scenario.get("min_aircraft_count") or self.max_aircraft_count
        # 0 (or absent) keeps every generated training scenario.
        self._max_initial_deviation_s = scenario.get("max_initial_deviation_s") or 0
        self._scenario_rejected = False
        # Failed training seeds are re-drawn from a bounded per-environment queue; they were
        # already drawn from this reservation-guarded stream, so replay never reaches reserved worlds.
        # The queue is not part of the boundary state: each segment or phase restarts it empty.
        self._failure_replay_fraction = (
            0.0 if self.evaluation_mode else scenario.get("failure_replay_fraction") or 0.0
        )
        self._failed_seeds: deque[int] = deque(maxlen=FAILED_SEED_CAPACITY)
        self._replay_rng = random.Random(identity)
        self._scenario_replayed = False
        self.max_steps_per_episode = scenario["episode_steps"]
        self.forecast_min_lookahead_s = scenario.get("forecast_min_lookahead_s") or 0
        self.scenario_start_time_offset_max_s = scenario["start_offset_max_s"]
        self.success_time_dev_s = payload["thresholds"][
            "operational_success_s" if self.evaluation_mode else "training_success_s"
        ]
        self._training_payload = payload
        # The verified immutable sections are consumed by this instance.
        self.consumed_config_fields = (
            "root_seed",
            "topology",
            "reserved_manifest",
            "reserved_conditions.physics",
            "reserved_conditions.native_artifact",
            "scenario",
            "thresholds.operational_success_s"
            if self.evaluation_mode
            else "thresholds.training_success_s",
            "thresholds.lockout_s",
            "thresholds.auto_solve_s",
            "features.dimensions",
            "features.linear_scale_s",
            "features.deviation_encoding",
            "features.knee_s",
            "features.cap_s",
            "features.neighbor_selection",
            "reward.max_for_zero_delay",
            "reward.max_for_zero_infringement",
            "reward.action_penalty_weight",
            "reward.speed_penalty_weight",
            "reward.global_share",
            "reward.success_bonus",
            "reward.realized_failure_cost",
            "reward.unresolved_failure_cost",
            "reward.success_margin_bonus_max",
            "reward.delayed_clean_bonus",
            "reward.count_baseline",
            "reward.action_damping_steps",
            "reward.speed_damping_steps",
        )
        if self.evaluation_mode:
            self.consumed_config_fields += ("thresholds.training_success_s",)
        if payload["reserved_conditions"]["version"] == 2:
            self.consumed_config_fields += (
                "reserved_conditions.benchmark.early_auto_solve_target_window_s",
            )
        runtime_print(
            "RunConfig",
            "consumed",
            semantic_digest=self.consumed_semantic_digest,
            fields=self.consumed_config_fields,
            stream=identity,
        )

    def capture_boundary_state(self) -> dict:
        """A closed configured outcome, or pristine guarded env, with no live continuation."""
        from copy import deepcopy

        if not self._has_reset_once:
            if self.simulator.world is not None:
                raise ValueError("pristine scenario boundary has a native world")
            audit = deepcopy(getattr(self, "_checkpoint_previous_audit", None))
            if audit is None and self.get_training_stream_state()["stream"]["cursor"] != 0:
                raise ValueError("pristine scenario boundary has consumed draws")
        else:
            if (
                not self._scenario_finished
                or self.last_end_reason not in vocabulary.ALL_END_REASONS
            ):
                raise ValueError("scenario has not reached a configured final outcome")
            if self.simulator.world is None or self.prev_world is None:
                raise ValueError("completed scenario lacks native/planning audit state")
            audit = {
                "seed": self.scenario_seed,
                "last_end_reason": self.last_end_reason,
                "physically_complete": self.outcome_state.physically_complete,
                "world_sha256": content_digest(json.loads(self.simulator.world.json())),
                "planning_world_sha256": content_digest(json.loads(self.prev_world.json())),
                "policy_steps": self.step_counter,
            }
        return {
            "version": 1,
            "boundary": "pristine" if audit is None else "closed",
            "stream": self.get_training_stream_state(),
            "audit": audit,
        }

    def restore_boundary_state(self, state: dict, *, validate_only=False, transition=None) -> None:
        """Restore exact stream identity before first reset; keep this segment's journal."""
        from copy import deepcopy

        if self._has_reset_once or self.simulator.world is not None or self._runtime_activated:
            raise ValueError("restore scenario boundary before activation and first reset")
        validate_environment_envelope(state)
        audit = state["audit"]
        own = self.get_training_stream_state()
        saved = state["stream"]
        if transition is not None:
            transition.validate_stream_envelope(saved, own)
        elif set(saved) != set(own) or any(
            saved[key] != own[key]
            for key in ("version", "semantic_digest", "reservation_sha256", "conditions_sha256")
        ):
            raise ValueError("scenario boundary stream identity mismatch")
        stream = deepcopy(self._training_stream)
        assert stream is not None
        stream.set_state(saved["stream"])
        if not validate_only:
            # The phase transition validated the outer contracts; the stream itself
            # retains its exact domain, root, coordinates and next draw.
            assert self._training_stream is not None
            self._training_stream.set_state(saved["stream"])
            self._checkpoint_previous_audit = deepcopy(audit)

    def _resolve_episode_max_aircraft_count(self) -> int:
        """Training draws the count from the scenario seed; evaluation keeps the full count."""
        if self.evaluation_mode:
            return int(self.max_aircraft_count)
        return random.Random(self.scenario_seed).randint(
            self._min_aircraft_count, self.max_aircraft_count
        )

    def _next_training_scenario_seed(self) -> int:
        """A recently failed training seed with probability failure_replay_fraction, else a new draw."""
        self._remember_failure()
        self._scenario_replayed = bool(self._failed_seeds) and (
            self._replay_rng.random() < self._failure_replay_fraction
        )
        if self._scenario_replayed:
            return self._failed_seeds.popleft()
        return super()._next_training_scenario_seed()

    def _remember_failure(self) -> None:
        if not self._failure_replay_fraction or not self._scenario_finished:
            return
        if not self._scenario_rejected and self.last_outcome.kind not in vocabulary.SUCCESS_KINDS:
            self._failed_seeds.append(self.scenario_seed)

    def reset(self, *, seed=None, options=None):
        """Training redraws from the seed stream while the pre-autosolve deviation exceeds the cap."""
        result = super().reset(seed=seed, options=options)
        while self._scenario_rejected:
            result = super().reset(options=options)
        return result

    def _apply_initial_metric_baselines(self, raw_metrics, policy_start_metrics) -> None:
        super()._apply_initial_metric_baselines(raw_metrics, policy_start_metrics)
        cap = 0 if self.evaluation_mode else self._max_initial_deviation_s
        self._scenario_rejected = bool(cap) and any(
            abs(deviation_s) > cap for deviation_s in raw_metrics.deviations_by_aircraft.values()
        )

    def _journal_training_scenario(self, status: str) -> None:
        if self._training_stream is None:
            return
        if status == "consumed" and self._scenario_rejected:
            status = "rejected"
        assert self._scenario_artifact_path is not None and self._training_payload is not None
        event = {
            **self.get_training_stream_state(),
            "status": status,
            "seed": self.scenario_seed,
            "replayed": self._scenario_replayed,
            "requested_airport": self._airport,
            "requested_aircraft_count": self.episode_max_aircraft_count,
            "scenario_config_sha256": self._training_payload["scenario"]["config_sha256"],
            "consumed_fields": self.consumed_config_fields,
        }
        if variant := self._training_payload["scenario"].get("generator_variant"):
            event["generator_variant"] = variant
        if status in {"generated", "consumed"}:
            assert self.simulator.world is not None
            event["world_sha256"] = content_digest(json.loads(self.simulator.world.json()))
            event["generation_attempts"] = self.simulator.last_scenario_generation_attempts
            event["aircraft_count"] = len(self.simulator.world.aircraft)
            event["observed_airports"] = sorted(
                {
                    airport
                    for aircraft in self.simulator.world.aircraft
                    if (
                        airport := detect_airport_from_route(
                            aircraft.route, self.settings.core.action.custom_action_mask_by_airport
                        )
                    )
                    is not None
                }
            )
        if status == "consumed" and self.prev_world is not None:
            event["planning_world_sha256"] = content_digest(json.loads(self.prev_world.json()))
        append_scenario_event(self._scenario_artifact_path, event)

    def _reserved_seeds(self) -> frozenset[int]:
        assert self._training_payload is not None
        return effective_reserved_seeds(self._training_payload["reserved_manifest"])

    def _check_contract_before_reset(self) -> None:
        payload = self._training_payload
        if payload is not None:
            require_initial_preprocessing_window(
                payload["thresholds"], payload["reserved_conditions"]
            )
