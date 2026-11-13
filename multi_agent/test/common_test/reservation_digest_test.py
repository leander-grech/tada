from pathlib import Path

from common.hashing import content_digest
from common.jsonio import read_json

RESERVATION = Path(__file__).resolve().parents[2] / "config/contracts/bgy/reserved-v1.json"


def test_recorded_reservation_digest_is_reproduced():
    body = read_json(RESERVATION)
    claimed = body.pop("manifest_sha256")
    assert content_digest(body) == claimed
