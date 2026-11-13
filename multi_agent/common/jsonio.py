"""Single JSON file reader."""

import json
from pathlib import Path

_REQUIRED = object()


def read_json(path: Path, default: object = _REQUIRED):
    """Parse a JSON file; a missing file yields `default` when one is given."""
    path = Path(path)
    if default is not _REQUIRED and not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
