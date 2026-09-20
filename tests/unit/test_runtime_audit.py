"""The runtime's hash-chained audit: tamper evidence, redaction and the shared file format."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_supervision_tree.agentic.audit import GENESIS_HASH, verify_file
from agent_supervision_tree.core.canonical import canonical_json
from agent_supervision_tree.runtime.audit import AuditChain

pytestmark = pytest.mark.unit


def fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_entries_chain_from_the_genesis_hash() -> None:
    chain = AuditChain(clock=fixed_clock)
    assert chain.head == GENESIS_HASH
    first = chain.append("root/w0", "actor.restarted", {"child": "w0"})
    second = chain.append("root", "supervisor.escalated", {"child": "w0"})
    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.hash
    assert chain.head == second.hash
    assert len(chain) == 2
    assert first.seq == 0


def test_max_entries_is_validated_and_bounds_memory() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        AuditChain(max_entries=0)
    chain = AuditChain(max_entries=2, clock=fixed_clock)
    for index in range(5):
        chain.append("root", "actor.restarted", {"index": index})
    # The chain keeps growing (seq and head), only the retained window is bounded.
    assert len(chain) == 5
    assert len(chain.recent(100)) == 2


def test_data_is_redacted_before_it_is_hashed(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path, clock=fixed_clock)
    entry = chain.append("root/sink", "approval.decided", {"note": "call bob@example.com"})
    chain.flush()
    assert "bob@example.com" not in path.read_text(encoding="utf-8")
    assert "[REDACTED:EMAIL]" in canonical_json(entry.data)
    chain.close()
    assert verify_file(path).ok


def test_the_written_file_verifies_with_the_shared_verifier(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path, clock=fixed_clock)
    for index in range(5):
        chain.append("root", "actor.restarted", {"index": index})
    chain.close()
    result = verify_file(path, expect_head=chain.head)
    assert result.ok
    assert result.entries == 5
    assert result.head == chain.head


def test_a_tampered_line_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path, clock=fixed_clock)
    chain.append("root", "actor.restarted", {"child": "w0"})
    chain.append("root", "actor.restarted", {"child": "w1"})
    chain.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"w0"', '"w9"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = verify_file(path)
    assert not result.ok
    assert result.error is not None


def test_counting_and_filtering_by_action() -> None:
    chain = AuditChain(clock=fixed_clock)
    chain.append("root", "actor.restarted", {})
    chain.append("root", "actor.restarted", {})
    chain.append("root", "supervisor.escalated", {})
    assert chain.count("actor.restarted") == 2
    assert [entry.action for entry in chain.recent(10, action="supervisor.escalated")] == [
        "supervisor.escalated"
    ]
    assert chain.path is None


def test_export_writes_the_retained_entries(tmp_path: Path) -> None:
    chain = AuditChain(clock=fixed_clock)
    for index in range(3):
        chain.append("root", "actor.restarted", {"index": index})
    path = tmp_path / "exported.jsonl"
    assert chain.export_jsonl(path) == 3
    assert verify_file(path).ok
