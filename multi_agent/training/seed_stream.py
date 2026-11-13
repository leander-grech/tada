"""Bounded deterministic scenario draws with explicit reserved-set exclusion.

A cursor counts every draw, including rejected reservations. Hashing separates
stream identities; it does not promise globally unique training scenarios.
"""

from __future__ import annotations

from common.hashing import content_digest, nonnegative_int

SEED_MAX = 2**31 - 1
ALGORITHM = "sha256-json-counter-v1"


class ScenarioSeedStream:
    """One independently restorable (phase, runner, vector) scenario stream."""

    def __init__(
        self,
        root: int,
        stream_id: tuple[int, int, int],
        reserved_seeds: frozenset[int],
        cursor: int = 0,
        *,
        domain: str = "training",
    ) -> None:
        if domain not in {"training", "operational_evaluation"}:
            raise ValueError("unsupported scenario stream domain")
        self.domain = domain
        self.root = nonnegative_int(root, "root")
        if len(stream_id) != 3:
            raise ValueError("stream_id must contain phase, runner and vector")
        self.stream_id = tuple(nonnegative_int(item, "stream_id") for item in stream_id)
        self.reserved_seeds = frozenset(reserved_seeds)
        for seed in self.reserved_seeds:
            if nonnegative_int(seed, "reserved seed") > SEED_MAX:
                raise ValueError("reserved seed exceeds native signed integer range")
        if len(self.reserved_seeds) == SEED_MAX + 1:
            raise ValueError("reserved seeds exhaust the scenario domain")
        self.reserved_manifest_hash = content_digest(sorted(self.reserved_seeds))
        self.cursor = nonnegative_int(cursor, "cursor")
        self.rejected = 0

    def next_seed(self) -> int:
        """Consume draws until a seed outside the reservation is found."""
        while True:
            identity = [ALGORITHM, self.root, self.stream_id, self.cursor]
            if self.domain != "training":
                identity = ["sha256-json-domain-counter-v2", self.domain, *identity[1:]]
            digest = content_digest(identity)
            candidate = int(digest[:16], 16) & SEED_MAX
            self.cursor += 1
            if candidate not in self.reserved_seeds:
                return candidate
            self.rejected += 1

    def get_state(self) -> dict:
        """Return JSON-serializable state including the complete stream identity."""
        return {
            "version": 2,
            "domain": self.domain,
            "algorithm": ALGORITHM
            if self.domain == "training"
            else "sha256-json-domain-counter-v2",
            "root": self.root,
            "stream_id": list(self.stream_id),
            "reserved_manifest_hash": self.reserved_manifest_hash,
            "cursor": self.cursor,
            "rejected": self.rejected,
        }

    @staticmethod
    def validate_state_structure(state: dict) -> tuple[int, int]:
        """Validate a saved cursor without constructing or changing a live stream."""
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "version",
                "domain",
                "algorithm",
                "root",
                "stream_id",
                "reserved_manifest_hash",
                "cursor",
                "rejected",
            }
            or type(state["version"]) is not int
            or state["version"] != 2
            or state["domain"] not in ("training", "operational_evaluation")
            or state["algorithm"]
            != (ALGORITHM if state["domain"] == "training" else "sha256-json-domain-counter-v2")
        ):
            raise ValueError("invalid scenario stream identity or state")
        nonnegative_int(state["root"], "root")
        if not isinstance(state["stream_id"], list) or len(state["stream_id"]) != 3:
            raise ValueError("invalid scenario stream coordinates")
        for coordinate in state["stream_id"]:
            nonnegative_int(coordinate, "stream_id")
        digest = state["reserved_manifest_hash"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid scenario stream reservation hash")
        cursor = nonnegative_int(state["cursor"], "cursor")
        rejected = nonnegative_int(state["rejected"], "rejected")
        if rejected > cursor:
            raise ValueError("rejected draws cannot exceed cursor")
        return cursor, rejected

    def set_state(self, state: dict) -> None:
        """Validate every field before changing any live stream state."""
        cursor, rejected = self.validate_state_structure(state)
        own = self.get_state()
        if set(state) != set(own) or any(
            state[key] != own[key]
            for key in (
                "version",
                "domain",
                "algorithm",
                "root",
                "stream_id",
                "reserved_manifest_hash",
            )
        ):
            raise ValueError("scenario stream identity mismatch")
        self.cursor, self.rejected = cursor, rejected
