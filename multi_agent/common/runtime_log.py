"""Helpers for consistent runtime print formatting."""

import os
import sys
from typing import Any, TextIO

RESET = "\x1b[0m"
GREEN = "\x1b[1;32m"
YELLOW = "\x1b[1;33m"
RED = "\x1b[1;31m"
CYAN = "\x1b[1;36m"
WHITE = "\x1b[1;37m"
STATUS_COLOR_FIELDS = {"cond", "conditions", "status", "gate"}

EVENT_COLORS = {
    "error": RED,
    "warning": YELLOW,
    "ready": GREEN,
    "advanced": GREEN,
    "synced": GREEN,
    "done": GREEN,
    "checkpoint": GREEN,
    "override": YELLOW,
    "restore": CYAN,
    "decision": CYAN,
    "conditions": CYAN,
}
LABEL_COLORS = {
    "success": GREEN,
    "infringement": RED,
    "delayed": YELLOW,
    "timeout": RED,
}


def force_runtime_log_color() -> None:
    """Mirror the trainer default by forcing runtime log color when unset."""
    os.environ.setdefault("RUNTIME_LOG_COLOR", "always")


def colorize_runtime_label(
    label: str,
    *,
    stream: TextIO | None = None,
    color_mode: str | None = None,
) -> str:
    """Apply shared semantic coloring to summary labels when color output is enabled."""
    target = sys.stdout if stream is None else stream
    if not _should_use_color(target, color_mode=_resolve_color_mode(color_mode)):
        return label
    return _colorize_value(label)


def _colorize_event(event: str, event_name: str) -> str:
    color = EVENT_COLORS.get(event_name, WHITE)
    return f"{color}{event}{RESET}"


def _colorize_value(value: str) -> str:
    """Apply semantic coloring to known label values and status tokens."""
    label_color = LABEL_COLORS.get(value.strip().lower())
    if label_color is not None:
        return f"{label_color}{value}{RESET}"
    return value


def _colorize_status_tokens(key: str, value: str) -> str:
    if key not in STATUS_COLOR_FIELDS:
        return value

    value = value.replace("(PASS)", f"{GREEN}(PASS){RESET}")
    value = value.replace("(FAIL)", f"{RED}(FAIL){RESET}")
    return value


def _resolve_color_mode(color_mode: str | None) -> str:
    mode = (color_mode or os.getenv("RUNTIME_LOG_COLOR", "auto")).strip().lower()
    return mode if mode in {"auto", "always", "never"} else "auto"


def _should_use_color(stream: TextIO, *, color_mode: str) -> bool:
    if color_mode == "always":
        return True
    if color_mode == "never" or os.getenv("NO_COLOR") is not None:
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _format_runtime_event(
    component: str,
    event: str,
    fields: dict[str, Any],
    *,
    use_color: bool,
) -> str:
    """Build a standardized runtime log line.

    Format:
        >>> [Component] EVENT: key=value | key=value
    """
    event_upper = event.strip().upper()
    event_name = event.strip().lower()
    rendered_event = _colorize_event(event_upper, event_name) if use_color else event_upper
    prefix = f">>> [{component}] {rendered_event}"
    entries = []
    for key, value in fields.items():
        if value is None:
            continue
        str_value = str(value)
        display_value = _colorize_value(str_value) if use_color else str_value
        entry = f"{key}={display_value}"
        if use_color:
            entry = _colorize_status_tokens(key, entry)
        entries.append(entry)

    if not entries:
        return prefix
    return f"{prefix}: {' | '.join(entries)}"


def runtime_print(
    component: str,
    event: str,
    *,
    flush: bool = True,
    file: TextIO | None = None,
    color_mode: str | None = None,
    **fields: Any,
) -> None:
    """Print a runtime line using the shared format."""
    stream = file if file is not None else sys.stdout
    use_color = _should_use_color(stream, color_mode=_resolve_color_mode(color_mode))
    print(
        _format_runtime_event(component, event, fields, use_color=use_color),
        flush=flush,
        file=file,
    )
