"""Risk categories preserve opening threats and deterministic top-K selection."""

import importlib.util
import itertools
import math

from config.settings import settings_from_config


def ranking():
    assert importlib.util.find_spec("models.neighbor_ranking"), "risk ranking helper is missing"
    from models.neighbor_ranking import NeighborRisk, neighbor_priority

    return NeighborRisk, neighbor_priority


def test_receding_candidates_cannot_crowd_out_approach():
    risk, priority = ranking()
    receding = [risk(str(i), 0, False, False, -i - 1, 20, 0, 20) for i in range(5)]
    approaching = risk("approach", 0, False, False, 100, 1, 0, 30)
    assert sorted([*receding, approaching], key=priority)[0] == approaching
    near_opening = risk("near", 0, False, True, -10, 2, 0.5, 2)
    assert sorted([approaching, near_opening], key=priority)[0] == near_opening


def test_severe_crowding_and_tied_permutations_are_stable():
    risk, priority = ranking()
    risks = [risk(name, 0.9, True, False, 500, 2, 0.5, 40) for name in "FEDCBA"]
    for permuted in itertools.permutations(risks):
        assert [r.callsign for r in sorted(permuted, key=priority)[:5]] == list("ABCDE")
    strongest = risk("Z", 0.95, True, False, 900, 10, 2, 100)
    assert sorted([*risks, strongest], key=priority)[0] == strongest


def test_physical_relevance_and_unclamped_extreme_cpa_order():
    risk, priority = ranking()
    near_cpa = risk("Z", 0, False, False, 1e20, 1, 0, 20)
    far_cpa = risk("A", 0, False, False, 1, 10, 0, 20)
    assert priority(near_cpa) < priority(far_cpa)
    earlier = risk("Z", 0, False, False, 1e10, 1, 0, 20)
    later = risk("A", 0, False, False, 1e20, 1, 0, 20)
    assert priority(earlier) < priority(later)
    vertically_irrelevant = risk("A", 0, False, False, 1, 0, 1.1, 20)
    assert priority(later) < priority(vertically_irrelevant)
    for raw in (-math.inf, -1e20, 0, math.inf, math.nan):
        stationary_or_past = risk("A", 0, False, False, raw, 20, 0, 20)
        assert priority(later) < priority(stationary_or_past)
    past_close_minimum = risk("A", 0, False, False, -10, 0, 0, 100)
    opening_nearby = risk("Z", 0, False, False, -20, 10, 0, 10)
    assert priority(opening_nearby) < priority(past_close_minimum)


def test_risk_membership_and_feature_encoding(native_base_json):
    import copy
    import json
    from dataclasses import replace
    from types import SimpleNamespace
    from typing import cast

    import numpy as np
    from flight_simulator import World

    from atc_env.env_config import observation_settings_from_config
    from config.config import Config
    from models.neighbors import _collect_neighbors
    from simulator.simulator import Simulator

    data = json.loads(native_base_json)
    ego = data["aircraft"][0]
    ego["under_control"] = True
    ego["state"]["position_m"] = [0, 0, 9000 * 0.3048]
    ego["state"]["velocity_m_s"] = [0, 100, 0]
    neighbors = []
    for i in range(6):
        neighbor = copy.deepcopy(ego)
        neighbor["call_sign"] = f"N{i}"
        neighbor["state"]["position_m"][1] = (20 + i) * 1852
        neighbor["state"]["velocity_m_s"][1] = 50 if i == 5 else 150
        neighbors.append(neighbor)
    data["aircraft"] = [ego, *neighbors]
    settings = replace(observation_settings_from_config(), max_neighbors=5)

    def collect(count=5):
        world = World.from_json(json.dumps(data))
        return _collect_neighbors(
            cast(Simulator, SimpleNamespace(world=world)),
            world.get_aircraft_by_callsign(ego["call_sign"]),
            {},
            replace(settings, max_neighbors=count),
            settings_from_config().observation,
        )

    corrected, mask = collect()
    # Only N5 has future CPA: it must survive truncation despite five past-CPA zeros.
    assert corrected[0, 12] > -1 and mask.sum() == 5
    assert corrected.dtype == np.float32
    all_rows, all_mask = collect(6)
    assert all_mask.sum() == 6
    assert {row.tobytes() for row in corrected} <= {row.tobytes() for row in all_rows}
    # Protective-band opening traffic beats the distant approaching candidate.
    neighbors[0]["state"]["position_m"][1] = 0.9 * Config.HORIZONTAL_SAFEZONE_RADIUS_NM * 1852
    neighbors[0]["state"]["position_m"][2] += 0.9 * Config.VERTICAL_SAFEZONE_RADIUS_FT * 0.3048
    near, _ = collect()
    assert near[0, 12] == -1
    assert near[0, 0] < corrected[0, 0]
    neighbors[0]["state"]["position_m"][1] = Config.HORIZONTAL_SAFEZONE_RADIUS_NM * 1852
    boundary, _ = collect()
    assert boundary[0, 12] == -1, (
        "protective boundary uses physical range, not the epsilon denominator"
    )
    neighbors[0]["state"]["position_m"][1] *= 1.01
    outside, _ = collect()
    assert outside[0].tobytes() == corrected[0].tobytes()
    # Existing visibility filters apply even if excluded traffic is very close.
    neighbors[0]["under_control"] = False
    visible, _ = collect()
    assert visible[0].tobytes() == corrected[0].tobytes()
    # A tiny nonzero relative velocity still approaches at a large finite time.
    # Its encoded channel intentionally retains the historical stationary cutoff.
    neighbors[0]["under_control"] = True
    neighbors[0]["state"]["position_m"] = [0, 20 * 1852, 9000 * 0.3048]
    neighbors[5]["state"]["velocity_m_s"][1] = 100 - 0.0001852
    data["aircraft"] = [ego, neighbors[0], neighbors[5]]
    slow, slow_mask = collect()
    assert slow_mask.sum() == 2
    assert slow[0, 12] == 1 and slow[1, 12] == -1


def test_risk_selection_keeps_far_converging_traffic(native_base_json):
    """A converging approach remains visible among closer opening neighbors."""
    import copy
    import json
    from dataclasses import replace
    from types import SimpleNamespace
    from typing import cast

    from flight_simulator import World

    from atc_env.env_config import observation_settings_from_config
    from models.neighbors import _collect_neighbors
    from simulator.simulator import Simulator

    settings = replace(observation_settings_from_config(), neighbor_selection="risk")
    data = json.loads(native_base_json)
    ego = data["aircraft"][0]
    ego["under_control"] = True
    ego["state"]["position_m"] = [0, 0, 9000 * 0.3048]
    ego["state"]["velocity_m_s"] = [0, 100, 0]

    def neighbor(callsign, ahead_nm, speed_m_s):
        aircraft = copy.deepcopy(ego)
        aircraft["call_sign"] = callsign
        aircraft["state"]["position_m"][1] = ahead_nm * 1852
        aircraft["state"]["velocity_m_s"][1] = speed_m_s
        return aircraft

    # Faster traffic ahead is opening; the slow far aircraft is overtaken head-on in
    # track, so its closest approach is at zero distance, below any separation minimum.
    opening = [neighbor(f"O{i}", 10 + 0.5 * i, 150) for i in range(settings.max_neighbors)]
    converging = neighbor("C", 25, 50)
    data["aircraft"] = [ego, *opening, converging]

    def collect(selection_settings):
        world = World.from_json(json.dumps(data))
        features, mask = _collect_neighbors(
            cast(Simulator, SimpleNamespace(world=world)),
            world.get_aircraft_by_callsign(ego["call_sign"]),
            {},
            selection_settings,
            settings_from_config().observation,
        )
        assert mask.sum() == settings.max_neighbors
        return features

    rows = collect(settings)
    # The approaching row is the only one with a future CPA (channel 12 above its floor).
    assert (rows[:, 12] > -1).sum() == 1
