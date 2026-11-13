"""ScenarioSourceMixin for ``ATCEnv``."""

from typing import TYPE_CHECKING, Any

from common.hashing import FrozenDict

if TYPE_CHECKING:
    from atc_env.environment import ATCEnv


class ScenarioSourceMixin:
    @staticmethod
    def _cantor_pair(non_negative_a: int, non_negative_b: int) -> int:
        """Return an injective pairing for non-negative integers."""
        total = non_negative_a + non_negative_b
        return (total * (total + 1)) // 2 + non_negative_b

    def _next_training_scenario_seed(self: "ATCEnv") -> int:
        if self._training_stream is None:
            raise ValueError("no reserved training stream configured")
        return self._training_stream.next_seed()

    def get_training_stream_state(self: "ATCEnv") -> dict:
        """Checkpoint identity and draw counters, independent of an active world."""
        if self._training_stream is None or self._training_payload is None:
            raise ValueError("no reserved training stream configured")
        return {
            "version": 1,
            "semantic_digest": self.consumed_semantic_digest,
            "reservation_sha256": self._training_payload["reserved_manifest"]["manifest_sha256"],
            "conditions_sha256": self._training_payload["reserved_conditions"]["conditions_sha256"],
            "stream": self._training_stream.get_state(),
        }

    def set_training_stream_state(self: "ATCEnv", state: dict) -> None:
        """Restore on a newly constructed environment before its first reset."""
        if self._has_reset_once:
            raise ValueError("restore training stream before the first reset")
        own = self.get_training_stream_state()
        if set(state) != set(own) or any(
            state[key] != own[key]
            for key in ("version", "semantic_digest", "reservation_sha256", "conditions_sha256")
        ):
            raise ValueError("training stream contract identity mismatch")
        assert self._training_stream is not None
        self._training_stream.set_state(state["stream"])

    def _next_evaluation_scenario_seed(self: "ATCEnv") -> int:
        """Return the next deterministic evaluation seed from the configured manifest."""
        stream = self._training_stream
        if stream is not None and not (self._inference_checkpoint and self._eval_seeds):
            self._eval_episode_index += 1
            return stream.next_seed()
        if not self._eval_seeds:
            raise ValueError("evaluation_seeds must contain at least one seed")

        self._eval_episode_index = self._eval_episode_index + 1
        fixed_count = len(self._eval_seeds)
        randomize_quota = max(0, int(self._eval_randomize_count))

        if self._eval_episode_index < fixed_count:
            return int(self._eval_seeds[self._eval_episode_index])

        if self._eval_randomize_used < randomize_quota:
            assert self._eval_random is not None
            eval_seed_set = set(self._eval_seeds)
            seed = next(
                candidate
                for candidate in iter(lambda: self._eval_random.randint(0, 2**31 - 1), None)
                if candidate not in eval_seed_set
            )
            self._eval_randomize_used += 1
            return int(seed)

        idx = self._eval_episode_index % fixed_count
        return int(self._eval_seeds[idx])

    def _select_evaluation_seed(self: "ATCEnv") -> int:
        """Advance the manifest cursor only after a finished scenario.

        A reset that arrives mid-scenario (sampler chunk cut, forced reset, or a
        reset immediately following another reset) replays the current seed
        instead of consuming the next manifest entry, so exact manifest coverage
        never depends on wrap-around laps.
        """
        current = self.scenario_seed
        if current is None or self._eval_episode_index < 0 or self._scenario_finished:
            return self._next_evaluation_scenario_seed()
        return int(current)

    @property
    def evaluation_scenario_count(self: "ATCEnv") -> int:
        """Number of evaluation scenarios this env instance is responsible for.

        With the manifest sharded across env runners this is the size of *this*
        runner's slice, which is exactly how many scenarios it must sample to
        cover its share of the manifest once. A scenario may span several RLlib
        episodes as aircraft enter and leave.
        """
        return len(self._eval_seeds)

    def set_next_world(self: "ATCEnv", world: Any) -> None:
        """Queue a World object to be used on the next reset() call.

        The pending world is consumed (cleared) when reset() runs.
        Used by InferenceEngine to inject externally-created scenarios.
        """
        if not self.evaluation_mode or (
            self._training_payload is not None and not self._inference_checkpoint
        ):
            raise ValueError("reserved training scenarios cannot use injected worlds")
        if self._inference_checkpoint:
            from common.world_digest import world_digest

            if len(world.arrival_targets) != self.max_aircraft_count:
                raise ValueError("injected world target count differs from saved scenario contract")
            self.inference_input_binding = FrozenDict(
                {
                    "purpose": "caller_owned_serving_request",
                    "world_sha256": world_digest(world),
                    "semantic_digest": self.consumed_semantic_digest,
                }
            )
        self._pending_world = world

    def _resolve_episode_max_aircraft_count(self: "ATCEnv") -> int:
        """Return the fixed aircraft count for the upcoming episode."""
        return int(self.max_aircraft_count)

    def _authorize_evaluation_seeds(self: "ATCEnv", seeds) -> None:
        """Authorize explicit evaluation seeds; a standalone env reserves nothing to refuse.

        Inference envs override this with the evaluation seed authorizer.
        """
