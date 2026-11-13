"""Validate reserved role populations, generation order and frozen conditions."""

import csv
import io
from pathlib import Path

from common.hashing import bytes_sha256, content_digest, nonnegative_int
from common.jsonio import read_json
from training.config_schema import exact_keys, positive, require_sha256
from training.seed_stream import ALGORITHM, SEED_MAX, ScenarioSeedStream

CONTRACT_ROOT = Path(__file__).resolve().parents[1] / "config/contracts"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _validate_strata(strata, campaign):
    if not isinstance(strata, (list, tuple)) or not strata:
        raise ValueError("reservation strata are required")
    ids = set()
    for stratum in strata:
        exact_keys(
            stratum,
            {"id", "airport", "difficulty", "aircraft_count", "weight", "scenario_config_sha256"},
            "stratum",
        )
        for field in ("id", "airport", "difficulty"):
            _text(stratum[field], f"stratum {field}")
        if stratum["id"] in ids or stratum["id"] == "published-validation":
            raise ValueError("stratum identities must be distinct")
        ids.add(stratum["id"])
        positive(stratum["aircraft_count"], "stratum aircraft_count", integer=True)
        positive(stratum["weight"], "stratum weight")
        require_sha256(stratum["scenario_config_sha256"], "stratum configuration")
    if campaign and (
        len(strata) != 1 or strata[0]["aircraft_count"] != 10 or strata[0]["weight"] != 1.0
    ):
        raise ValueError("campaign requires its single common-capacity primary stratum")
    frozen = (conditions["benchmark"]["stratum"] for conditions in _contract_conditions("*"))
    if campaign and (strata[0]["airport"], strata[0]["difficulty"]) not in {
        (stratum["airport"], stratum["difficulty"])
        for stratum in frozen
        if stratum["scenario_config_sha256"] == strata[0]["scenario_config_sha256"]
    }:
        raise ValueError("campaign stratum airport/difficulty differ from its frozen conditions")
    return ids


def _contract_conditions(version):
    """Every frozen conditions document of a version in the contract input tree."""
    pattern = f"conditions*-{version}.json"
    for path in sorted([*CONTRACT_ROOT.glob(pattern), *CONTRACT_ROOT.glob(f"*/{pattern}")]):
        yield read_json(path)


def _validate_generation(manifest):
    generation = manifest["generation"]
    exact_keys(
        generation,
        {"algorithm", "root", "stream_id", "cursor", "reserved_rejections", "duplicate_rejections"},
        "reservation generation",
    )
    if generation["algorithm"] != ALGORITHM or list(generation["stream_id"]) != [0, 0, 0]:
        raise ValueError("unsupported reservation generation identity")
    for field in ("root", "cursor", "reserved_rejections", "duplicate_rejections"):
        nonnegative_int(generation[field], field)
    validation = {row["seed"] for row in manifest["validation"]}
    stream = ScenarioSeedStream(generation["root"], (0, 0, 0), frozenset(validation))
    seen = set(validation)
    expected = []
    candidates = [row["seed"] for group in ("heldout", "replacements") for row in manifest[group]]
    while len(expected) < len(candidates):
        seed = stream.next_seed()
        if seed not in seen:
            expected.append(seed)
            seen.add(seed)
    if (
        candidates != expected
        or generation["cursor"] != stream.cursor
        or generation["reserved_rejections"] != stream.rejected
        or generation["duplicate_rejections"] != stream.cursor - stream.rejected - len(expected)
    ):
        raise ValueError("reservation candidate order or generation counters mismatch")


def validate_reservation_manifest(manifest: dict) -> frozenset[int]:
    """Validate meaning as well as content hashes before excluding seed roles."""
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if claimed != content_digest(body):
        raise ValueError("reserved manifest hash mismatch")
    if body.get("generation_sha256") != content_digest(body.get("generation")):
        raise ValueError("reservation generation hash mismatch")
    if (
        type(body.get("version")) is not int
        or body.get("version") != 1
        or body.get("purpose") not in {"fixture", "campaign"}
    ):
        raise ValueError("unsupported reserved manifest")
    if body.get("replacement_policy") != "ordered-technical-generation-failure-only":
        raise ValueError("unsupported replacement policy")
    campaign = body["purpose"] == "campaign"
    strata = _validate_strata(body.get("strata"), campaign)
    seeds = []
    for group, count in (("validation", 100), ("heldout", 200), ("replacements", 200)):
        rows = body.get(group)
        if not isinstance(rows, (list, tuple)) or not rows or (campaign and len(rows) != count):
            raise ValueError(f"invalid reserved {group} population")
        for row in rows:
            fields = {"seed", "stratum"} if group == "validation" else {"seed", "stratum", "status"}
            exact_keys(row, fields, f"reserved {group} row")
            if row["stratum"] not in (
                {"published-validation"} if group == "validation" else strata
            ):
                raise ValueError("reservation references unknown stratum")
            if group != "validation" and row["status"] != "unmaterialized":
                raise ValueError("frozen reservation entries must remain unmaterialized")
            seed = nonnegative_int(row["seed"], "reserved seed")
            if seed > SEED_MAX:
                raise ValueError("reserved seed exceeds native range")
            seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError("reservation must be duplicate-free")
    if campaign:
        published = body.get("published_validation")
        if not isinstance(published, dict):
            raise ValueError("campaign published validation evidence is required")
        exact_keys(published, {"path", "sha256"}, "published validation")
        if published["path"] != "config/evaluation_seed_manifest.csv":
            raise ValueError("campaign published validation path mismatch")
        data = (CONTRACT_ROOT.parent / "evaluation_seed_manifest.csv").read_bytes()
        rows = list(csv.DictReader(io.StringIO(data.decode())))
        if published["sha256"] != bytes_sha256(data) or [
            row["seed"] for row in body["validation"]
        ] != [int(row["seed"]) for row in rows]:
            raise ValueError("campaign must reserve the ordered published validation identities")
    _validate_generation(body)
    return frozenset(seeds)


def _validate_campaign_evidence(body):
    source = body["baseline_source"]
    exact_keys(
        source,
        {"commit", "environment_ast_sha256", "generation_preprocessing", "note"},
        "baseline source",
    )
    commit = source["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise ValueError("baseline source commit must be a Git SHA")
    require_sha256(source["environment_ast_sha256"], "baseline environment source")
    _text(source["note"], "baseline source note")
    preprocessing = source["generation_preprocessing"]
    exact_keys(preprocessing, {"sources", "referenced_settings"}, "baseline preprocessing")
    sources = preprocessing["sources"]
    if not isinstance(sources, dict) or not {
        "simulator/simulator.py",
        "simulator/auto_solve.py",
        "atc_env/environment.py",
    } <= set(sources):
        raise ValueError("baseline preprocessing sources are incomplete")
    for identity in sources.values():
        require_sha256(identity, "baseline preprocessing source")
    settings = preprocessing["referenced_settings"]
    if not isinstance(settings, dict) or "AUTO_SOLVE_TARGET_WINDOW_S" not in settings:
        raise ValueError("baseline initial preprocessing settings are incomplete")
    for expression in settings.values():
        _text(expression, "baseline referenced setting")
    native = body["native_artifact"]
    exact_keys(
        native,
        {
            "mode",
            "version",
            "source_identity",
            "source_provenance",
            "wheel_sha256",
            "extension_sha256",
        },
        "baseline native artifact",
    )
    if native["mode"] not in {"external", "source"}:
        raise ValueError("campaign native evidence cannot use a fixture")
    _text(native["version"], "native version")
    _text(native["source_provenance"], "native provenance")
    for field in ("wheel_sha256", "extension_sha256"):
        require_sha256(native[field], field)
    if native["mode"] == "source":
        require_sha256(native["source_identity"], "native source identity")


def _previous_conditions(digest):
    """The version 1 conditions a campaign amendment links to, wherever it is archived."""
    for conditions in _contract_conditions("v1"):
        body = {key: value for key, value in conditions.items() if key != "conditions_sha256"}
        if conditions["conditions_sha256"] == digest == content_digest(body):
            return conditions
    raise ValueError("previous conditions lineage mismatch")


def validate_reserved_conditions(conditions: dict, manifest: dict) -> None:
    """Bind campaign metadata, immutable lineage and resolved preprocessing."""
    body = dict(conditions)
    claimed = body.pop("conditions_sha256", None)
    if claimed != content_digest(body):
        raise ValueError("reserved conditions hash mismatch")
    campaign = manifest["purpose"] == "campaign"
    fields = {
        "version",
        "purpose",
        "reservation_sha256",
        "physics",
        "benchmark",
        "baseline_source",
        "native_artifact",
    }
    if body.get("version") == 2:
        fields.add("previous_conditions_sha256")
    exact_keys(body, fields, "reserved conditions")
    if (
        type(body["version"]) is not int
        or body["version"] not in {1, 2}
        or body["purpose"] != manifest["purpose"]
        or body["reservation_sha256"] != manifest["manifest_sha256"]
    ):
        raise ValueError("conditions reservation identity mismatch")
    if campaign and body["version"] != 2:
        raise ValueError("campaign requires complete version 2 conditions")
    exact_keys(
        body["physics"],
        {"horizontal_safezone_radius_nm", "vertical_safezone_radius_ft", "simulator_timestep_ms"},
        "reserved physics",
    )
    for key, value in body["physics"].items():
        positive(value, key, integer=key == "simulator_timestep_ms")
    benchmark = body["benchmark"]
    fields = {"stratum", "episode_steps", "start_offset_max_s", "preprocessing"}
    if body["version"] == 2:
        fields.add("early_auto_solve_target_window_s")
        positive(benchmark.get("early_auto_solve_target_window_s"), "early auto-solve window")
        require_sha256(body["previous_conditions_sha256"], "previous conditions")
    exact_keys(benchmark, fields, "benchmark conditions")
    if benchmark["stratum"] not in manifest["strata"]:
        raise ValueError("conditions stratum is not in the frozen reservation")
    positive(benchmark["episode_steps"], "benchmark episode_steps", integer=True)
    positive(
        benchmark["start_offset_max_s"], "benchmark start_offset_max_s", zero=True, integer=True
    )
    _text(benchmark["preprocessing"], "benchmark preprocessing")
    if not isinstance(body["baseline_source"], dict) or not body["baseline_source"]:
        raise ValueError("baseline source evidence is required")
    native = body["native_artifact"]
    if not isinstance(native, dict) or not native.get("mode"):
        raise ValueError("baseline native artifact evidence is required")
    if native["mode"] == "external" and native.get("source_identity") is not None:
        raise ValueError("external native provenance cannot claim a source identity")
    if campaign:
        _validate_campaign_evidence(body)
        previous = _previous_conditions(body["previous_conditions_sha256"])
        previous.pop("conditions_sha256")
        if (
            any(
                body[key] != previous[key]
                for key in ("purpose", "reservation_sha256", "physics", "native_artifact")
            )
            or {
                key: value
                for key, value in benchmark.items()
                if key != "early_auto_solve_target_window_s"
            }
            != previous["benchmark"]
        ):
            raise ValueError("conditions amendment cannot change frozen benchmark conditions")


FROZEN_BGY_RESERVATION = CONTRACT_ROOT / "bgy/reserved-v1.json"


def effective_reserved_seeds(manifest):
    """Scenario streams exclude the run's own reservation and the frozen BGY benchmark."""
    frozen = validate_reservation_manifest(read_json(FROZEN_BGY_RESERVATION))
    return validate_reservation_manifest(manifest) | frozen
