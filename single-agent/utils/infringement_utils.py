"""Utility functions for handling infringements and separation calculations"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from config.config import Config
from flight_simulator import Infringement, World




@dataclass
class InfringementTuple:
    """One predicted/realised loss of separation, as seen from one aircraft.

    Carries the pair's closeness on BOTH bands, because the reward and the observation
    deliberately disagree about how far away is far enough:

    * ``value``     — severity on the 3-5 NM band. What the pair is *charged* for, and the
                      only thing the tier gate and episode termination look at. 0 beyond 5 NM.
    * ``proximity`` — closeness on the wide 3-10 NM detection band. What the policy *sees*.
                      Non-zero out to 10 NM so a conflict can be watched developing.
    """

    value: float
    timestamp: float
    counterpart: str
    proximity: float = 0.0
    horizontal_nm: float = 0.0
    vertical_ft: float = 0.0


def normalize_separation(
    horizontal_nm: float,
    vertical_ft: float,
    horizontal_safezone: float,
    horizontal_conflict: float,
    vertical_safezone: float,
    vertical_conflict: float,
    shape_p: float = 1.0,
    _min=min,
    _max=max,
) -> float:
    """
    Normalize separation values to a 0-1 scale where:
    1.0 = separation is at or below conflict threshold
    0.0 = separation is at or above safezone threshold

    ``shape_p`` bends the decay across the band: the linear normalisation ``u`` is raised to
    ``u ** shape_p``. p = 1.0 is the linear ramp; p > 1 is convex, concentrating the gradient
    just outside the conflict radius and flattening the approach to the safezone edge.

    The shape is applied ONCE, to the combined value, rather than per axis. That is exactly
    equivalent — any monotonically increasing map commutes with ``min`` — and it is one call
    instead of two.
    """

    # Precompute reciprocals once per call to avoid repeated divisions
    inv_h_range = 1.0 / (horizontal_safezone - horizontal_conflict)
    inv_v_range = 1.0 / (vertical_safezone - vertical_conflict)

    # Raw normalized values (unclamped)
    h = (horizontal_safezone - horizontal_nm) * inv_h_range
    v = (vertical_safezone - vertical_ft) * inv_v_range

    # Clamp to [0.0, 1.0] using C-implemented min/max (branch-light)
    h = _max(0.0, _min(1.0, h))
    v = _max(0.0, _min(1.0, v))

    # Combine axes via minimum: both axes needs to be bad for actual infringement
    combined = _min(h, v)

    if shape_p != 1.0 and 0.0 < combined < 1.0:
        combined = combined ** shape_p

    return float(combined)


def pair_severity(horizontal_nm: float, vertical_ft: float) -> float:
    """Severity [0,1] of a pair at this separation, on the SEVERITY band.

    1.0 means a realised loss of separation: at or inside both
    ``Config.SEVERITY_CONFLICT_NM`` (3 NM) and ``Config.VERTICAL_CONFLICT_RADIUS_FT`` (500 ft).
    0.0 means legal spacing — at or outside ``Config.SEVERITY_SAFEZONE_NM`` (5 NM) horizontally,
    or outside the vertical safezone.

    This is the single definition used by the reward, the tier gate and episode termination.
    It is deliberately narrower than the DETECTION band used by :func:`pair_proximity`.
    """
    return normalize_separation(
        abs(float(horizontal_nm)),
        abs(float(vertical_ft)),
        Config.SEVERITY_SAFEZONE_NM,
        Config.SEVERITY_CONFLICT_NM,
        Config.VERTICAL_SAFEZONE_RADIUS_FT,
        Config.VERTICAL_CONFLICT_RADIUS_FT,
        shape_p=float(Config.SEVERITY_SHAPE_P),
    )


def pair_proximity(horizontal_nm: float, vertical_ft: float) -> float:
    """Proximity [0,1] of a pair on the wide DETECTION band — observation use only.

    Saturates at 1.0 at the loss-of-separation boundary (3 NM) and decays linearly to 0 at
    ``Config.DETECTION_SAFEZONE_RADIUS_NM`` (10 NM). This is what the policy *sees*, and it is
    intentionally wider than what the policy is *charged for*: the agent needs to watch a
    conflict develop from 10 NM out even though closing to 6 NM costs it nothing.
    """
    return normalize_separation(
        abs(float(horizontal_nm)),
        abs(float(vertical_ft)),
        Config.DETECTION_SAFEZONE_RADIUS_NM,
        Config.DETECTION_CONFLICT_RADIUS_NM,
        Config.VERTICAL_SAFEZONE_RADIUS_FT,
        Config.VERTICAL_CONFLICT_RADIUS_FT,
    )


def is_realised_los(severity: float) -> bool:
    """True if this severity counts as a realised loss of separation (<= 3 NM AND <= 500 ft).

    The severity ramp is clamped, so a real LoS lands on exactly 1.0; the epsilon guards
    against float round-trips through the simulator boundary rather than admitting near-misses.
    """
    return float(severity) >= float(Config.LOS_SEVERITY_THRESHOLD) - 1e-9




def get_world_infringements(
    world: Optional[World],
    current_time_s: Optional[float] = None,
    horizon_time_s: Optional[int] = None,
) -> Dict[str, List[InfringementTuple]]:
    """
    Get normalized infringement values for all aircraft in the world.
    Returns a dict mapping aircraft call signs to a list of :class:`InfringementTuple`,
    each carrying both the charged severity (``value``) and the wider observation
    proximity (``proximity``).

    An entry is kept when it is visible on the DETECTION band, not when it is chargeable,
    so pairs between 5 and 10 NM still reach the observation with ``value == 0``.
    """
    if world is None:
        return {}

    result: Dict[str, List[InfringementTuple]] = {}

    def _iter_relevant_infringements():
        for infr in world.infringements:
            if not isinstance(infr, Infringement.LOSS_OF_SEPARATION):  # type: ignore[arg-type]
                continue

            pair = [str(x) for x in getattr(infr, "aircraft_pair", [])]
            ts = getattr(infr, "timestamp_s", None)
            if not pair or ts is None:
                continue
            if current_time_s is not None and horizon_time_s is not None:
                within = (
                    float(current_time_s) <= ts <= (float(current_time_s) + float(horizon_time_s))
                )
                if not within:
                    continue
            yield infr, pair, ts

    for infr, pair, ts in _iter_relevant_infringements():
        # Both values are symmetric for the two aircraft in the pair.
        h_nm = abs(float(infr.horizontal_separation_nm))  # type: ignore[attr-defined]
        v_ft = abs(float(infr.vertical_separation_ft))  # type: ignore[attr-defined]

        proximity = pair_proximity(h_nm, v_ft)
        if proximity <= 0.0:
            continue

        value = pair_severity(h_nm, v_ft)

        for cs in pair:
            counterpart = next((p for p in pair if p != cs), cs)
            result.setdefault(cs, []).append(
                InfringementTuple(
                    value=float(value),
                    timestamp=float(ts),
                    counterpart=str(counterpart),
                    proximity=float(proximity),
                    horizontal_nm=h_nm,
                    vertical_ft=v_ft,
                )
            )

    # Sort each aircraft's infringements by timestamp ascending
    for cs, infringements in result.items():
        infringements.sort(key=lambda vt: vt.timestamp)

    return result
