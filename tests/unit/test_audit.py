from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_supervision_tree.agentic.audit import (
    AuditLog,
    anchor_path_for,
    append_anchor,
    main,
    read_anchors,
    verify_file,
)
from agent_supervision_tree.agentic.errors import AuditIntegrityError
from agent_supervision_tree.core.canonical import canonical_json, sha256_hex

pytestmark = pytest.mark.unit


def fixed_clock() -> Callable[[], datetime]:
    return lambda: datetime(2026, 1, 1, tzinfo=UTC)


def write_log(path: Path, entries: int = 5) -> AuditLog:
    log = AuditLog(path, clock=fixed_clock())
    for index in range(entries):
        log.append("agent:test", "step", {"index": index, "note": f"entry {index}"})
    return log


def lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def test_a_valid_chain_verifies(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    result = verify_file(path, expect_head=log.head)
    assert result.ok
    assert result.entries == 5
    assert result.head == log.head


def test_modified_entry_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path)
    lines = lines_of(path)
    entry = json.loads(lines[2])
    entry["data"]["index"] = 99
    lines[2] = canonical_json(entry) + "\n"
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    result = verify_file(path)
    assert (result.ok, result.error, result.line) == (False, "entry content modified", 3)


def test_rehashed_entry_breaks_the_chain_at_the_next_entry(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path)
    lines = lines_of(path)
    entry = json.loads(lines[2])
    entry["data"]["index"] = 99
    entry["hash"] = sha256_hex(canonical_json({k: v for k, v in entry.items() if k != "hash"}))
    lines[2] = canonical_json(entry) + "\n"
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    result = verify_file(path)
    assert (result.ok, result.error, result.line) == (False, "broken hash chain", 4)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda lines: lines[:2] + lines[3:], "sequence gap or reordering"),
        (lambda lines: [lines[0], lines[2], lines[1], *lines[3:]], "sequence gap or reordering"),
        (
            lambda lines: [*lines[:1], lines[1].replace(":", ": ", 1), *lines[2:]],
            "non-canonical encoding",
        ),
        (lambda lines: [*lines[:1], "{not json\n", *lines[2:]], "malformed entry"),
        (lambda lines: [*lines[:-1], lines[-1].rstrip("\n")], "truncated final line"),
    ],
)
def test_deletion_reordering_reencoding_and_truncation_are_detected(
    tmp_path: Path, mutate: Callable[[list[str]], list[str]], error: str
) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path)
    path.write_text("".join(mutate(lines_of(path))), encoding="utf-8", newline="\n")
    result = verify_file(path)
    assert not result.ok
    assert result.error == error


def test_tail_truncation_needs_an_anchored_head(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    head = log.head
    path.write_text("".join(lines_of(path)[:-1]), encoding="utf-8", newline="\n")
    assert verify_file(path).ok
    anchored = verify_file(path, expect_head=head)
    assert not anchored.ok
    assert anchored.error is not None
    assert "head mismatch" in anchored.error


@given(data=st.data())
def test_any_single_byte_change_is_detected(data: st.DataObject) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        write_log(path, entries=4)
        raw = bytearray(path.read_bytes())
        index = data.draw(st.integers(min_value=0, max_value=len(raw) - 1))
        original = raw[index]
        raw[index] = data.draw(st.integers(0, 255).filter(lambda value: value != original))
        path.write_bytes(bytes(raw))
        assert not verify_file(path).ok


def test_reopening_continues_the_chain_and_refuses_a_tampered_log(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path, entries=2)
    reopened = AuditLog(path, clock=fixed_clock())
    entry = reopened.append("agent:test", "step", {"index": 2})
    assert entry.seq == 2
    assert verify_file(path).entries == 3

    path.write_text(
        path.read_text(encoding="utf-8").replace("entry 0", "entry X"),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(AuditIntegrityError, match="modified"):
        AuditLog(path)


def test_entries_are_redacted_before_hashing(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("agent:test", "tool.executed", {"email": "jane@example.com", "password": "x"})
    text = path.read_text(encoding="utf-8")
    assert "jane@example.com" not in text
    assert '"password":"[REDACTED]"' in text
    assert verify_file(path).ok


def test_two_writers_on_one_file_extend_a_single_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path, clock=fixed_clock())
    second = AuditLog(path, clock=fixed_clock())
    first.append("agent:a", "step", {"n": 1})
    second.append("agent:b", "step", {"n": 2})
    first.append("agent:a", "step", {"n": 3})
    result = verify_file(path, expect_head=first.head)
    assert (result.ok, result.entries) == (True, 3)


def test_truncation_while_open_is_detected_on_the_next_append(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path, entries=3)
    path.write_text("".join(lines_of(path)[:2]), encoding="utf-8", newline="\n")
    with pytest.raises(AuditIntegrityError, match="truncated or rolled back"):
        log.append("agent:test", "step", {"index": 3})
    assert verify_file(path).entries == 2


def test_history_rewritten_while_open_is_detected_on_the_next_append(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path, entries=2)
    forged = tmp_path / "forged.jsonl"
    forger = AuditLog(forged, clock=fixed_clock())
    for index in range(3):
        forger.append("agent:forger", "step", {"index": index})
    path.write_bytes(forged.read_bytes())
    with pytest.raises(AuditIntegrityError, match="rewritten"):
        log.append("agent:test", "step", {"index": 2})


@pytest.mark.parametrize("value", [False, 0.0])
def test_type_coerced_fields_fail_verification_even_with_a_recomputed_hash(
    tmp_path: Path, value: object
) -> None:
    # A lax validator reads false or 0.0 as seq 0, so one entry would have several valid encodings.
    path = tmp_path / "audit.jsonl"
    write_log(path, entries=1)
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["seq"] = value
    entry["hash"] = sha256_hex(canonical_json({k: v for k, v in entry.items() if k != "hash"}))
    path.write_text(canonical_json(entry) + "\n", encoding="utf-8", newline="\n")
    result = verify_file(path)
    assert (result.ok, result.error, result.line) == (False, "malformed entry", 1)


def test_missing_log_does_not_verify(tmp_path: Path) -> None:
    result = verify_file(tmp_path / "missing.jsonl")
    assert (result.ok, result.error) == (False, "audit log not found")


def test_verify_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    assert main(["verify", str(path), "--expect-head", log.head]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    path.write_text(
        path.read_text(encoding="utf-8").replace("entry 1", "entry 7"),
        encoding="utf-8",
        newline="\n",
    )
    assert main(["verify", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["line"] == 2


# --- head anchoring -------------------------------------------------------------------------


def test_an_anchor_catches_a_truncation_that_the_chain_alone_accepts(tmp_path: Path) -> None:
    """A truncated tail leaves a self-consistent prefix; only a recorded head notices."""
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    append_anchor(anchor_path_for(path), entries=5, head=log.head)
    path.write_text("".join(lines_of(path)[:-2]), encoding="utf-8", newline="\n")
    assert verify_file(path).ok  # the prefix verifies on its own
    result = verify_file(path, anchor=anchor_path_for(path))
    assert not result.ok
    assert result.error is not None
    assert "shorter than its anchor" in result.error


def test_an_anchor_catches_a_forged_entry_appended_after_the_real_head(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    anchor = anchor_path_for(path)
    append_anchor(anchor, entries=5, head=log.head)
    log.append("agent:attacker", "step", {"index": 99, "note": "forged"})
    assert verify_file(path).ok  # a forged continuation is a valid chain
    assert verify_file(path, anchor=anchor).ok  # ... and appending is allowed
    # But rewriting history behind the anchor is not.
    lines = lines_of(path)
    path.write_text("".join(lines[:3] + lines[5:]), encoding="utf-8", newline="\n")
    result = verify_file(path, anchor=anchor)
    assert not result.ok


def test_an_anchor_that_does_not_match_the_prefix_is_a_rewritten_history(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path)
    anchor = anchor_path_for(path)
    append_anchor(anchor, entries=3, head="f" * 64)
    result = verify_file(path, anchor=anchor)
    assert not result.ok
    assert result.error is not None
    assert "anchor mismatch" in result.error


def test_anchors_round_trip_and_an_empty_anchor_file_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    anchor = anchor_path_for(path)
    assert read_anchors(anchor) == []
    assert verify_file(path, anchor=anchor).ok
    append_anchor(anchor, entries=2, head=log.head)
    assert [(item.entries, item.head) for item in read_anchors(anchor)] == [(2, log.head)]


def test_a_corrupt_anchor_file_fails_verification(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    write_log(path)
    anchor = anchor_path_for(path)
    anchor.write_text("not json\n", encoding="utf-8", newline="\n")
    result = verify_file(path, anchor=anchor)
    assert not result.ok
    assert result.error is not None
    assert "anchor" in result.error


def test_verify_cli_uses_the_sibling_anchor_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "audit.jsonl"
    log = write_log(path)
    append_anchor(anchor_path_for(path), entries=5, head=log.head)
    assert main(["verify", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["anchors"] == 1
    path.write_text("".join(lines_of(path)[:-1]), encoding="utf-8", newline="\n")
    assert main(["verify", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    # ... and an operator can still ask for the chain alone.
    assert main(["verify", str(path), "--no-anchor"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
