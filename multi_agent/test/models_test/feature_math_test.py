"""Scalar feature math: scheduling, geometry, route bounds, deviation encoding, normalization."""

import math

import numpy as np
import pytest

from atc_env.env_config import observation_settings_from_config
from config.config import Config
from config.settings import settings_from_config
from models import feature_transforms, neighbors, observation_features, observations
from models.observation_settings import ObservationSettings
from training.config_defaults import FEATURE_DEFAULTS


def test_signed_time_and_bounded_mapping():
    from models.feature_transforms import bounded_signed, signed_time_remaining

    assert signed_time_remaining(600, 45) == 555
    assert signed_time_remaining(30, 45) == -15
    assert signed_time_remaining(45, 45) == 0
    assert bounded_signed(2, 3) == pytest.approx(0.5)
    values = [bounded_signed(x, 3) for x in [-10000, -2, 0, 2, 10000]]
    assert -1 < values[0] < values[1] < values[2] < values[3] < values[4] < 1
    assert values == pytest.approx([-v for v in reversed(values)])
    for invalid in [0, -1, math.nan, math.inf]:
        with pytest.raises(ValueError, match="scale"):
            bounded_signed(1, invalid)
        with pytest.raises(ValueError, match="gain"):
            bounded_signed(1, 1, invalid)


@pytest.mark.parametrize(
    "minimum,maximum,expected",
    [
        (6000, 10000, 8000),
        (6000, None, 6000),
        (None, 8000, 8000),
        (None, None, 9000),
        (None, 0, 0),
        (0, None, 0),
        (0, 0, 0),
    ],
)
def test_route_altitude_bounds(minimum, maximum, expected):
    from models.feature_transforms import representative_altitude

    assert representative_altitude(minimum, maximum, 9000) == expected


def test_inverted_route_bounds_are_rejected():
    from models.feature_transforms import representative_altitude

    with pytest.raises(ValueError, match="altitude"):
        representative_altitude(9000, 6000, 8000)


def test_compass_ego_basis_and_stationary_north_convention():
    from models.feature_transforms import ego_components

    assert ego_components(0, 10, 0) == pytest.approx((10, 0))
    assert ego_components(10, 0, math.pi / 2) == pytest.approx((10, 0))
    assert ego_components(10, 0, 0) == pytest.approx((0, 10))
    for vx, vy in [(0, 0), (-0.0, -0.0), (1e-8, -1e-8)]:
        assert feature_transforms._heading_rad(vx, vy) == 0


def test_common_rotation_preserves_ego_features_and_raw_cpa():
    from models.feature_transforms import ego_components

    # Clockwise common rotation in the east/north coordinate plane.
    theta = 0.731

    def rotate(x: float, y: float) -> tuple[float, float]:
        return math.cos(theta) * x + math.sin(theta) * y, -math.sin(theta) * x + math.cos(theta) * y

    pos, vel, heading = (3, 10), (-0.02, -0.1), 0.27
    ego_pos, ego_vel = ego_components(*pos, heading), ego_components(*vel, heading)
    np.testing.assert_allclose(ego_components(*rotate(*pos), heading + theta), ego_pos)
    np.testing.assert_allclose(ego_components(*rotate(*vel), heading + theta), ego_vel)
    assert feature_transforms.compute_ttca_dcpa(
        *ego_pos, *ego_vel, Config.TTCA_MAX_S
    ) == pytest.approx(feature_transforms.compute_ttca_dcpa(*pos, *vel, Config.TTCA_MAX_S))


def encoder():
    encode = getattr(feature_transforms, "encode_deviation", None)
    assert callable(encode), "selectable deviation encoding must be implemented"
    return encode


@pytest.mark.parametrize("mode", ["linear", "signed_log"])
def test_encoding_zero_oddness_monotonicity_and_saturation(mode):
    encode = encoder()

    def f(value):
        return encode(value, mode=mode, linear_scale_s=180, knee_s=20, cap_s=180)

    assert f(0) == 0
    assert f(180) == f(1e200) == 1
    assert f(-180) == f(-1e200) == -1
    values = [f(value) for value in [0, 1, 10, 20, 60, 120, 180]]
    assert all(left < right for left, right in zip(values, values[1:]))
    for value in (1, 17, 90, 180, 2000):
        assert f(-value) == -f(value)
    assert f(20) == pytest.approx(math.log10(2) if mode == "signed_log" else 1 / 9)
    assert abs(f(20 + 1e-6) - f(20 - 1e-6)) < 1e-6
    assert f(math.inf) == 1 and f(-math.inf) == -1
    assert math.isnan(f(math.nan))


def test_invalid_encoding_parameters_are_rejected():
    encode = encoder()
    kwargs = dict(mode="signed_log", linear_scale_s=180, knee_s=20, cap_s=180)
    for field in ("linear_scale_s", "knee_s", "cap_s"):
        for value in (0, -1, math.nan, math.inf):
            with pytest.raises(ValueError):
                encode(10, **{**kwargs, field: value})
    with pytest.raises(ValueError):
        encode(10, **{**kwargs, "mode": "unknown"})
    with pytest.raises(ValueError):
        encode(10, **{**kwargs, "knee_s": 200})


def test_selected_options_are_instance_local():
    features = {
        **FEATURE_DEFAULTS,
        "deviation_encoding": "signed_log",
        "neighbor_selection": "risk",
        "linear_scale_s": 180,
        "knee_s": 20,
        "cap_s": 180,
    }
    settings = ObservationSettings.from_features(features)
    assert settings.encode_deviation(20) == pytest.approx(math.log10(2))
    assert settings.neighbor_selection == "risk"
    assert ObservationSettings.from_features(
        {**features, "deviation_encoding": "linear"}
    ).encode_deviation(20) == pytest.approx(1 / 9)


def test_scalar_clamp_matches_numpy_scalar_types_nonfinite_and_bounds():
    clamp = getattr(feature_transforms, "scalar_clamp", None)
    assert callable(clamp), "scalar_clamp is the shared scalar clipping contract"
    for scalar_type in (float, np.float32, np.float64, int, np.int32, np.int64):
        values: list[int | float] = [-3, -1, 0, 1, 3]
        if scalar_type in (float, np.float32, np.float64):
            values += [-0.0, 0.123456789, -math.inf, math.inf, math.nan]
        for value in values:
            for low, high in (
                (-1.0, 1.0),
                (-0.123456789, 0.123456789),
                (2.0, -2.0),
                (0.0, 0.0),
                (-math.inf, math.inf),
                (math.nan, 1),
                (-1, math.nan),
            ):
                expected = float(np.clip(scalar_type(value), low, high))
                actual = clamp(scalar_type(value), low, high)
                assert type(actual) is float
                if math.isnan(expected):
                    assert math.isnan(actual)
                else:
                    assert actual == expected
                    if expected == 0:
                        assert math.copysign(1, actual) == math.copysign(1, expected)


def test_normalizers_and_cpa_preserve_reference_nonfinite_and_degenerate_semantics():
    from config.config import Config

    for scalar_type in (float, np.float32, np.float64):
        for raw in (-math.inf, -123.456, -0.0, 0.25, 4000, math.inf, math.nan):
            value = scalar_type(raw)
            for low, high in ((-3, 7), (2, 2), (7, -3)):
                expected = (
                    0.5 if high == low else float(np.clip((value - low) / (high - low), 0, 1))
                )
                np.testing.assert_equal(feature_transforms._normalize(value, low, high), expected)
            for scale in (-1, 0, 7):
                expected = 0.0 if scale <= 0 else float(np.clip(value / scale, -1, 1))
                np.testing.assert_equal(feature_transforms._normalize_sym(value, scale), expected)
    for dx, dy, vx, vy in ((2, 0, 1, 0), (2, 0, -1, 0), (2, 3, 0, 0), (1e20, 0, -1, 0)):
        speed2 = vx * vx + vy * vy
        time = (
            float(Config.TTCA_MAX_S)
            if speed2 < 1e-12
            else float(np.clip(-(dx * vx + dy * vy) / speed2, 0, Config.TTCA_MAX_S))
        )
        assert feature_transforms.compute_ttca_dcpa(dx, dy, vx, vy, Config.TTCA_MAX_S) == (
            time,
            math.hypot(dx + vx * time, dy + vy * time),
        )


def test_full_packed_features_retain_numpy_reference_bytes(monkeypatch):
    scalar = observation_features.AircraftScalarFeatures(
        234.12345, -234.8765, 0.5, -0.5, -77, 133.123, 1, 29.3
    )
    waypoint = observation_features.WaypointRelFeatures(
        4.12345, -8.23456, -3400, 9.205, -0.2, 0.8, 211.8, 1
    )
    neighbor = neighbors.NeighborFeatures(
        8.123,
        -4.77,
        1234,
        -0.0444,
        0.0678,
        512,
        -0.0034,
        9.7,
        0.3,
        0.95,
        -0.2,
        0.8,
        734,
        4.24,
        0.75,
        -0.0077,
    )

    def packed():
        return [
            scalar.to_numpy(observation_settings_from_config(), settings_from_config().observation),
            waypoint.to_numpy(settings_from_config().observation),
            *neighbors._pack_and_normalize(
                [neighbor], observation_settings_from_config(), settings_from_config().observation
            ),
            observations._global_context_vector([-1, 0.123456789, 2]),
        ]

    optimized = packed()
    monkeypatch.setattr(
        feature_transforms,
        "_normalize",
        lambda value, low, high: (
            0.5 if high == low else float(np.clip((value - low) / (high - low), 0, 1))
        ),
    )
    monkeypatch.setattr(
        feature_transforms,
        "_normalize_sym",
        lambda value, scale: 0.0 if scale <= 0 else float(np.clip(value / scale, -1, 1)),
    )
    reference = packed()
    for actual, expected in zip(optimized, reference, strict=True):
        assert actual.dtype == expected.dtype == np.float32
        assert actual.tobytes() == expected.tobytes()
