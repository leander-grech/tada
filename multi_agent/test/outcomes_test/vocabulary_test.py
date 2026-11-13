from outcomes import vocabulary as v


def test_kind_to_end_reason_is_total_over_kinds():
    assert set(v.END_REASON_BY_KIND) == set(v.ALL_KINDS)


def test_eligible_kinds_are_the_ones_mapping_to_selectable_end_reasons():
    assert {v.END_REASON_BY_KIND[k] for k in v.ELIGIBLE_KINDS} == v.SELECTABLE_END_REASONS


def test_persisted_literals_are_unchanged():
    assert v.ALL_END_REASONS == ("success", "infringement", "timeout", "delayed", "unresolved")
    assert v.PHYSICAL_SUCCESS == "physical_success"
