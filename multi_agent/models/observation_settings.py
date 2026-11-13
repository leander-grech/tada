"""Immutable implemented observation settings."""

from collections.abc import Mapping
from dataclasses import dataclass

from config.observation_fields import AIRCRAFT_SCALAR_FIELDS, GLOBAL_CONTEXT_FIELDS
from models.feature_transforms import encode_deviation


@dataclass(frozen=True)
class ObservationSettings:
    linear_scale_s: float
    max_waypoints: int
    max_neighbors: int
    waypoint_dim: int
    neighbor_dim: int
    deviation_encoding: str = "linear"
    neighbor_selection: str = "risk"
    knee_s: float = 60.0
    cap_s: float = 900.0

    def __post_init__(self):
        self.encode_deviation(0.0)
        if self.neighbor_selection != "risk":
            raise ValueError("unsupported neighbor selection")

    def encode_deviation(self, value_s: float) -> float:
        return encode_deviation(
            value_s,
            mode=self.deviation_encoding,
            linear_scale_s=self.linear_scale_s,
            knee_s=self.knee_s,
            cap_s=self.cap_s,
        )

    @classmethod
    def from_features(cls, features: Mapping):
        dims = features["dimensions"]
        if (
            dims["aircraft_scalar"] != len(AIRCRAFT_SCALAR_FIELDS)
            or dims["global_context"] != len(GLOBAL_CONTEXT_FIELDS)
            or dims["waypoint"] != 8
            or dims["neighbor"] != 16
        ):
            raise ValueError("feature dimensions do not match the implemented observation schema")
        return cls(
            float(features["linear_scale_s"]),
            int(dims["max_waypoints"]),
            int(dims["max_neighbors"]),
            int(dims["waypoint"]),
            int(dims["neighbor"]),
            features["deviation_encoding"],
            features["neighbor_selection"],
            float(features["knee_s"]),
            float(features["cap_s"]),
        )
