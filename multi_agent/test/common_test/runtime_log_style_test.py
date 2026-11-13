"""Tests for shared runtime print formatting."""

import io

from common.runtime_log import runtime_print


def _plain(component, event, **fields):
    stream = io.StringIO()
    runtime_print(component, event, file=stream, color_mode="never", **fields)
    return stream.getvalue().rstrip("\n")


def test_plain_runtime_print_includes_component_event_and_fields():
    line = _plain(
        "Readiness",
        "decision",
        trial="e8f70_00000",
        stage=0,
        ready="False",
    )

    assert line == ">>> [Readiness] DECISION: trial=e8f70_00000 | stage=0 | ready=False"


def test_plain_runtime_print_skips_none_fields():
    line = _plain(
        "Readiness",
        "decision",
        trial="e8f70_00000",
        evidence=None,
        threshold="NO",
    )

    assert line == ">>> [Readiness] DECISION: trial=e8f70_00000 | threshold=NO"


def test_plain_runtime_print_without_fields():
    line = _plain("Trainer", "done")

    assert line == ">>> [Trainer] DONE"


def test_runtime_print_supports_colored_warning_and_status_tokens():
    stream = io.StringIO()

    runtime_print(
        "Readiness",
        "warning",
        file=stream,
        color_mode="always",
        cond="success=0.62>=0.60(PASS) | infringement=0.25<=0.20(FAIL)",
    )

    output = stream.getvalue()
    assert "\x1b[" in output
    assert "WARNING" in output
    assert "(PASS)" in output
    assert "(FAIL)" in output


def test_runtime_print_honors_always_color_override(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setenv("RUNTIME_LOG_COLOR", "always")

    runtime_print("Trainer", "done", file=stream)

    output = stream.getvalue()
    assert "\x1b[" in output


def test_plain_runtime_print_allows_color_as_regular_field():
    line = _plain("Trainer", "config", color="blue")

    assert line == ">>> [Trainer] CONFIG: color=blue"


def test_runtime_print_does_not_colorize_pass_fail_inside_message_field():
    stream = io.StringIO()

    runtime_print(
        "Readiness",
        "warning",
        file=stream,
        color_mode="always",
        message="literal(PASS) and literal(FAIL)",
    )

    output = stream.getvalue()
    message_part = output.split("message=", maxsplit=1)[1].strip()
    assert "\x1b[" not in message_part
    assert message_part == "literal(PASS) and literal(FAIL)"


def test_plain_runtime_print_allows_internal_flag_name_as_regular_field():
    line = _plain("Trainer", "config", __runtime_color="blue")

    assert line == ">>> [Trainer] CONFIG: __runtime_color=blue"
