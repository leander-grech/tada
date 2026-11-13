"""A phase loads a fresh guarded Algorithm through the complete bundle API."""

from dataclasses import replace
from pathlib import Path

import pytest

from actions.action_table import NUM_ACTIONS
from training.lifecycle import RunRequest
from training.phase import PHASE_FIELDS, PhaseTransition, apply_sets, extend_lineage
from training.run_config import reseal


def test_tune_load_uses_new_complete_bundle_algorithm(tmp_path, native_run, monkeypatch):
    from training import trainable

    calls = []
    old = type("Old", (), {"training_iteration": 0, "stop": lambda self: calls.append("stopped")})()
    new = object()

    def load(loaded, run, **kwargs):
        assert loaded == "inspected"
        assert run is native_run
        calls.append("loaded")
        return new

    monkeypatch.setattr(
        trainable,
        "inspect_checkpoint",
        lambda *args, **kwargs: calls.append("validated") or "inspected",
    )
    monkeypatch.setattr(
        trainable, "capture_boundary_state", lambda *args: calls.append("quiescent")
    )
    monkeypatch.setattr(trainable, "load_checkpoint_algorithm", load)

    trial = object.__new__(trainable.SingleRunTrainable)
    trial.algorithm = old
    trial.run = native_run
    trial.segment_dir = tmp_path
    trial.load_checkpoint("selected")
    assert trial.algorithm is new
    assert trial.hooks.algorithm is new and trial.hooks.segment_dir == tmp_path
    assert calls == ["validated", "quiescent", "stopped", "loaded"]


@pytest.fixture
def run(native_run):
    return native_run


def test_whitelist_maps_aliases_to_config_paths_with_types():
    assert PHASE_FIELDS == {
        "lr": ("optimizer", "learning_rate", float),
        "entropy": ("optimizer", "entropy_coeff", float),
        "train_batch_size": ("optimizer", "train_batch_size", int),
        "minibatch_size": ("optimizer", "minibatch_size", int),
        "aircraft_count": ("scenario", "aircraft_count", int),
        "min_aircraft_count": ("scenario", "min_aircraft_count", int),
        "max_initial_deviation_s": ("scenario", "max_initial_deviation_s", int),
        "failure_replay_fraction": ("scenario", "failure_replay_fraction", float),
        "forecast_min_lookahead_s": ("scenario", "forecast_min_lookahead_s", int),
        "action_mask": ("scenario", "custom_action_mask", str),
        "generator_variant": ("scenario", "generator_variant", str),
        "training_success_s": ("thresholds", "training_success_s", float),
        "unresolved_failure_cost": ("reward", "unresolved_failure_cost", float),
        "conflict_noop_cost": ("reward", "conflict_noop_cost", float),
        "conflict_weight": ("reward", "max_for_zero_infringement", float),
    }


def test_apply_changes_only_whitelisted_fields_and_records_lineage(run):
    phased, change = apply_sets(run, ["lr=0.0001", "aircraft_count=12"])
    assert phased.optimizer["learning_rate"] == 0.0001
    assert phased.scenario["aircraft_count"] == 12
    assert change == {
        "lr": [run.optimizer["learning_rate"], 0.0001],
        "aircraft_count": [run.scenario["aircraft_count"], 12],
    }


def test_non_whitelisted_field_is_rejected(run):
    with pytest.raises(ValueError, match="gamma"):
        apply_sets(run, ["gamma=0.9"])


def test_transition_rejects_any_other_difference(run):
    phased, changes = apply_sets(run, ["lr=0.0001"])
    phased = reseal(replace(phased, phase_id="next"), reward=dict(run.reward, count_baseline=99))
    with pytest.raises(ValueError, match="reward"):
        PhaseTransition.between(run, phased, changes)


def test_transition_rejects_unlisted_field_inside_a_whitelisted_section(run):
    phased, changes = apply_sets(run, ["lr=0.0001"])
    phased = reseal(replace(phased, phase_id="next"), optimizer=dict(phased.optimizer, gamma=0.5))
    with pytest.raises(ValueError, match="gamma"):
        PhaseTransition.between(run, phased, changes)


def test_transition_carries_learner_overrides_and_digests(run):
    phased, changes = apply_sets(replace(run, phase_id="next"), ["lr=0.0001", "aircraft_count=12"])
    transition = PhaseTransition.between(run, phased, changes)
    assert transition.overrides == {"learning_rate": 0.0001}
    assert (transition.source_digest, transition.target_digest) == (
        run.semantic_digest,
        phased.semantic_digest,
    )


def test_batch_sizes_phase_as_sampling_config_without_learner_overrides(run):
    batch = 2 * run.optimizer["train_batch_size"]
    sets = [f"train_batch_size={batch}", f"minibatch_size={batch // 4}"]
    phased, changes = apply_sets(replace(run, phase_id="next"), sets)
    transition = PhaseTransition.between(run, phased, changes)
    assert (phased.optimizer["train_batch_size"], phased.optimizer["minibatch_size"]) == (
        batch,
        batch // 4,
    )
    assert transition.overrides == {}
    with pytest.raises(ValueError, match="minibatch_size must divide"):
        apply_sets(run, [f"minibatch_size={batch // 4 + 1}"])


def test_phase_moves_inference_placement_but_not_the_learner_or_runner_topology(run):
    gpu = "GPU-00000000-0000-0000-0000-000000000001"
    saved = reseal(
        run,
        resources=dict(
            run.resources, inference_device="cuda", gpus_per_runner=0.075, inference_gpu_uuid=gpu
        ),
    )
    phased, changes = apply_sets(replace(run, phase_id="next"), ["lr=0.0001"])
    transition = PhaseTransition.between(saved, phased, changes)
    assert transition.changed == {
        "lr": changes["lr"],
        "resources.inference_device": ["cuda", phased.resources["inference_device"]],
        "resources.gpus_per_runner": [0.075, phased.resources["gpus_per_runner"]],
        "resources.inference_gpu_uuid": [gpu, phased.resources.get("inference_gpu_uuid")],
    }
    assert transition.overrides == {"learning_rate": 0.0001}
    for section, value in (
        ("resources", dict(phased.resources, cpus_per_learner=7.0)),
        ("topology", dict(phased.topology, env_runners=phased.topology["env_runners"] + 1)),
    ):
        with pytest.raises(ValueError, match=section):
            PhaseTransition.between(saved, reseal(phased, **{section: value}), changes)


def test_checkpoint_predating_min_aircraft_count_phases_with_a_fixed_count(run):
    saved = reseal(
        run,
        scenario={k: v for k, v in run.scenario.items() if k != "min_aircraft_count"},
    )
    phased, changes = apply_sets(replace(run, phase_id="next"), ["lr=0.0001"])
    assert PhaseTransition.between(saved, phased, changes).overrides == {"learning_rate": 0.0001}


def test_checkpoint_predating_the_unresolved_cost_phases_into_charging_it(run):
    saved = reseal(
        run, reward={k: v for k, v in run.reward.items() if k != "unresolved_failure_cost"}
    )
    phased, changes = apply_sets(replace(run, phase_id="next"), ["unresolved_failure_cost=3"])
    transition = PhaseTransition.between(saved, phased, changes)
    assert phased.reward["unresolved_failure_cost"] == 3.0
    assert transition.source_digest != transition.target_digest


@pytest.mark.parametrize(
    "mask", ["0" + "1" * (NUM_ACTIONS - 1), "1" * (NUM_ACTIONS - 1), "1" * (NUM_ACTIONS - 1) + "2"]
)
def test_action_mask_must_be_binary_per_action_and_allow_no_command(run, mask):
    with pytest.raises(ValueError, match="custom_action_mask"):
        apply_sets(replace(run, phase_id="next"), [f"action_mask={mask}"])


def test_action_mask_phase_binds_a_new_contract_without_learner_overrides(run):
    saved = reseal(
        run, scenario={k: v for k, v in run.scenario.items() if k != "custom_action_mask"}
    )
    assert saved.semantic_digest == run.semantic_digest
    phased, changes = apply_sets(
        replace(run, phase_id="next"), ["action_mask=" + "1" * (NUM_ACTIONS - 1) + "0"]
    )
    transition = PhaseTransition.between(saved, phased, changes)
    assert transition.source_digest != transition.target_digest
    assert transition.overrides == {}


def test_conflict_weight_phase_reweights_only_the_conflict_shaping(run):
    phased, changes = apply_sets(replace(run, phase_id="next"), ["conflict_weight=30"])
    transition = PhaseTransition.between(run, phased, changes)
    assert phased.reward == {**run.reward, "max_for_zero_infringement": 30.0}
    assert transition.source_digest != transition.target_digest
    assert transition.overrides == {}


def test_phase_requires_a_new_phase_name_and_same_run(run):
    same_phase, changes = apply_sets(run, ["lr=0.0001"])
    with pytest.raises(ValueError, match="phase name"):
        PhaseTransition.between(run, same_phase, changes)
    with pytest.raises(ValueError, match="phase name"):
        PhaseTransition.between(run, replace(same_phase, run_id="other", phase_id="next"), changes)


def test_resume_requires_identity(run):
    PhaseTransition.between(run, run, {}, mode="resume")
    phased, changes = apply_sets(run, ["lr=0.0001"])
    with pytest.raises(ValueError, match="resume"):
        PhaseTransition.between(run, phased, changes, mode="resume")


def test_lineage_accumulates_across_phases(run):
    first = extend_lineage(
        [], run, parent_sha="a" * 64, lifetime_iterations=5, changes={"lr": [1, 2]}
    )
    second = extend_lineage(first, run, parent_sha="b" * 64, lifetime_iterations=9, changes={})
    assert [e["parent_checkpoint_sha256"] for e in second] == ["a" * 64, "b" * 64]


def test_phase_requires_a_set_and_fresh_forbids_checkpoint():
    with pytest.raises(ValueError, match="phase"):
        RunRequest("phase", Path("c.json"), Path("x"), ())
    with pytest.raises(ValueError, match="fresh"):
        RunRequest("fresh", Path("c.json"), Path("x"), ())
    with pytest.raises(ValueError, match="resume"):
        RunRequest("resume", Path("c.json"), Path("x"), ("lr=0.1",))


def test_public_load_rejects_reported_segment_without_touching_algorithm(native_run, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from training import trainable

    calls = []
    trial = object.__new__(trainable.SingleRunTrainable)
    trial.run = native_run
    trial.algorithm = SimpleNamespace(training_iteration=1, stop=lambda: calls.append("stopped"))
    trial._iteration = 1
    trial._last_result = {"lifetime_iterations": 1}
    monkeypatch.setattr(
        trainable, "inspect_checkpoint", lambda *args, **kwargs: calls.append("inspected")
    )
    monkeypatch.setattr(
        trainable, "load_checkpoint_algorithm", lambda *args, **kwargs: calls.append("loaded")
    )
    with pytest.raises(ValueError, match="fresh.*unreported"):
        trial.load_checkpoint("selected")
    assert calls == []
