"""Pure scheduling and compass-coordinate observation transforms."""

import math

import numpy as np


def scalar_clamp(value: float, low: float, high: float) -> float:
    """Match float(np.clip(...)) while bypassing NumPy for Python scalars.

    NumPy scalar promotion is retained for its scalar types, including narrow
    floating-point bounds. Arrays still use vector clipping at their callers.
    """
    if any(type(item) not in (float, int) for item in (value, low, high)):
        return float(np.clip(value, low, high))
    if math.isnan(value) or math.isnan(low) or math.isnan(high):
        return math.nan
    if value <= low:
        value = low
    if value >= high:
        value = high
    return float(value)


def encode_deviation(
    value_s: float, *, mode: str, linear_scale_s: float, knee_s: float, cap_s: float
) -> float:
    """Encode seconds with a selectable odd, saturated linear or signed log map."""
    if mode not in {"linear", "signed_log"}:
        raise ValueError("unsupported deviation encoding")
    for name, value in (("linear_scale_s", linear_scale_s), ("knee_s", knee_s), ("cap_s", cap_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if knee_s > cap_s:
        raise ValueError("knee_s must not exceed cap_s")
    if mode == "linear":
        return scalar_clamp(value_s / linear_scale_s, -1.0, 1.0)
    magnitude = abs(value_s)
    if magnitude >= cap_s:
        return math.copysign(1.0, value_s)

    def log_ratio(value: float) -> float:
        ratio = value / knee_s
        return math.log1p(ratio) if math.isfinite(ratio) else math.log(value) - math.log(knee_s)

    return math.copysign(log_ratio(magnitude) / log_ratio(cap_s), value_s)


def signed_time_remaining(eta_s: float, live_age_s: float) -> float:
    """Seconds until the target ETA; overdue targets have negative remaining time."""
    return float(eta_s) - float(live_age_s)


def bounded_signed(value: float, scale: float, gain: float = 1.5) -> float:
    """Odd, monotonic arctan encoding with zero mapped to zero."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    return (2.0 / math.pi) * math.atan(gain * value / scale)


def ego_components(x: float, y: float, heading_rad: float) -> tuple[float, float]:
    """Project east/north components onto forward/right for atan2(vx, vy) heading."""
    sin_h, cos_h = math.sin(heading_rad), math.cos(heading_rad)
    return sin_h * x + cos_h * y, cos_h * x - sin_h * y


def representative_altitude(min_ft: float | None, max_ft: float | None, current_ft: float) -> float:
    """Represent a route constraint, preserving absent bounds and valid zero."""
    if min_ft is not None and max_ft is not None:
        if min_ft > max_ft:
            raise ValueError(f"inverted route altitude bounds: min {min_ft} ft > max {max_ft} ft")
        return (min_ft + max_ft) / 2.0
    if min_ft is not None:
        return float(min_ft)
    if max_ft is not None:
        return float(max_ft)
    return float(current_ft)


def _normalize(value, vmin, vmax) -> float:
    """Min-max normalization with clamping to [0, 1]."""
    if vmax == vmin:
        return 0.5
    return scalar_clamp((value - vmin) / (vmax - vmin), 0.0, 1.0)


def _normalize_sym(value, scale) -> float:
    """Map a value to [-1, 1] using symmetric scale with clamping."""
    if scale <= 0:
        return 0.0
    return scalar_clamp(value / scale, -1.0, 1.0)


def _map01_to_sym(x01) -> float:
    """Map [0,1] -> [-1,1]."""
    return float(2.0 * np.clip(x01, 0.0, 1.0) - 1.0)


def map_sym(value: float, half_range: float) -> float:
    """Helper: symmetric normalization to [-1, 1] with clamping."""
    return _normalize_sym(value, half_range)


def map_to_sym_unit(value: float, vmin: float, vmax: float) -> float:
    """Helper: min-max normalize to [0,1], then map to [-1,1]."""
    return _map01_to_sym(_normalize(value, vmin, vmax))


def _bounded_arctan01(value: float, scale: float, gain: float = 1.0) -> float:
    """Map non-negative value to [0,1) with bounded arctan scaling."""
    if scale <= 0.0:
        return 0.0
    ratio = max(0.0, float(value)) / float(scale)
    return float((2.0 / math.pi) * math.atan(max(1e-6, float(gain)) * ratio))


def map_positive_to_sym_bounded(value: float, scale: float, gain: float = 1.0) -> float:
    """Map non-negative value to [-1,1) with bounded arctan scaling."""
    return _map01_to_sym(_bounded_arctan01(value, scale, gain=gain))


def _heading_rad(vx: float, vy: float) -> float:
    """Compass heading radians of a horizontal velocity, atan2(vx, vy); north when stationary."""
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    return math.atan2(vx, vy)


def _bearing_sin_cos(dx, dy):
    """Return sin/cos of bearing from current pos to target (dx, dy)."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0, 1.0
    angle = math.atan2(dx, dy)
    return math.sin(angle), math.cos(angle)


def _distance(dx, dy):
    return float(math.hypot(dx, dy))


def compute_ttca_dcpa(dx_fwd, dy_right, dvx_nms, dvy_nms, ttca_max_s: float) -> tuple[float, float]:
    v2 = dvx_nms * dvx_nms + dvy_nms * dvy_nms
    if v2 < 1e-12:
        t_cpa = float(ttca_max_s)
    else:
        t_cpa = scalar_clamp(-(dx_fwd * dvx_nms + dy_right * dvy_nms) / v2, 0.0, ttca_max_s)
    d_cpa_nm = float(math.hypot(dx_fwd + dvx_nms * t_cpa, dy_right + dvy_nms * t_cpa))
    return t_cpa, d_cpa_nm
