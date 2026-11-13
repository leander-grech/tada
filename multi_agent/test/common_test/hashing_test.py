import pickle

import pytest

from common.hashing import (
    FrozenDict,
    bytes_sha256,
    canonical_json,
    content_digest,
    file_sha256,
    freeze_json,
    is_sha256,
    nonnegative_int,
)

SAMPLE = {"b": [1, 2.5, None, True], "a": {"z": "é", "y": []}}


def test_canonical_json_is_sorted_compact_ascii_escaped():
    assert canonical_json(SAMPLE) == '{"a":{"y":[],"z":"\\u00e9"},"b":[1,2.5,null,true]}'


def test_canonical_json_rejects_non_finite():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_content_digest_matches_known_vectors():
    # Fixed vectors protect the serialized content-digest contract.
    assert (
        content_digest(SAMPLE) == "cf384536d029d22b466b502efe292920811f376c9312ee42552c8c1823d669e0"
    )
    assert content_digest([]) == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"


def test_content_digest_ignores_key_order():
    assert content_digest({"a": 1, "b": 2}) == content_digest({"b": 2, "a": 1})


def test_bytes_and_file_sha256_agree(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == bytes_sha256(b"abc")
    assert (
        bytes_sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_is_sha256_accepts_only_lowercase_64_hex():
    good = "a" * 64
    assert is_sha256(good)
    assert not is_sha256(good.upper())
    assert not is_sha256(good[:-1])
    assert not is_sha256("g" * 64)
    assert not is_sha256(None)
    assert not is_sha256(b"a" * 64)


@pytest.mark.parametrize("bad", [-1, True, 1.0, "1", None])
def test_nonnegative_int_rejects_lossy_values(bad):
    with pytest.raises(ValueError, match="count must be a non-negative integer"):
        nonnegative_int(bad, "count")


def test_nonnegative_int_accepts_zero():
    assert nonnegative_int(0, "count") == 0


def test_frozen_dict_is_deeply_immutable_and_picklable():
    frozen = FrozenDict({"a": {"b": [1, {"c": 2}]}})
    assert frozen == {"a": {"b": (1, {"c": 2})}}
    assert isinstance(frozen["a"], FrozenDict)
    assert isinstance(frozen["a"]["b"][1], FrozenDict)
    for mutate in (
        lambda: frozen.__setitem__("x", 1),
        lambda: frozen["a"].update(x=1),
        lambda: frozen.pop("a"),
        lambda: frozen.clear(),
    ):
        with pytest.raises(TypeError, match="immutable"):
            mutate()
    assert pickle.loads(pickle.dumps(frozen)) == frozen
    assert frozen.copy() is frozen


def test_freeze_json_converts_lists_to_tuples():
    assert freeze_json([1, [2]]) == (1, (2,))
