"""The quarantine: bounded records, redaction, export and replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_supervision_tree.runtime.errors import QuarantineFullError
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.quarantine import Quarantine, QuarantineRecord

pytestmark = pytest.mark.unit


def envelope(index: int = 0, **payload: str) -> Envelope:
    return Envelope.create("doc.ingest", {"index": index, **payload})


def test_capacity_is_validated() -> None:
    with pytest.raises(ValueError, match="max_records"):
        Quarantine(0)


def test_a_record_keeps_identity_reason_and_attempts() -> None:
    quarantine = Quarantine(10)
    message = envelope(1).next_attempt()
    record = quarantine.add(message, reason="schema_violation", actor="root/w0")
    assert record.message_id == message.id
    assert (record.reason, record.actor, record.attempts) == ("schema_violation", "root/w0", 2)
    assert record.kind == "doc.ingest"
    assert record.replayable
    assert message.id in quarantine
    assert len(quarantine) == 1
    assert quarantine.reasons() == {"schema_violation": 1}


def test_payloads_are_redacted_before_they_are_stored() -> None:
    quarantine = Quarantine(10)
    message = Envelope.create("doc.ingest", {"text": "write to jane.doe@example.com now"})
    record = quarantine.add(message, reason="poison", actor="root/w0")
    assert isinstance(record.payload, dict)
    assert "jane.doe@example.com" not in str(record.payload["text"])
    assert "[REDACTED:EMAIL]" in str(record.payload["text"])


def test_a_non_json_payload_is_stored_opaquely_and_cannot_be_replayed() -> None:
    quarantine = Quarantine(10)
    control = Envelope(kind="llm.call", payload=object())
    record = quarantine.add(control, reason="unexpected_kind", actor="root/provider")
    assert record.payload == {"opaque": True}
    assert not record.replayable
    assert quarantine.take([control.id]) == []
    assert control.id in quarantine  # kept, because replaying it is impossible


def test_re_quarantining_the_same_id_replaces_the_record_without_growing() -> None:
    quarantine = Quarantine(2)
    message = envelope(1)
    quarantine.add(message, reason="first", actor="root/w0")
    quarantine.add(message.next_attempt(), reason="second", actor="root/w1")
    assert len(quarantine) == 1
    record = quarantine.get(message.id)
    assert record is not None
    assert (record.reason, record.attempts) == ("second", 2)
    assert quarantine.added_total == 2


def test_the_bound_is_enforced_and_overflow_is_counted() -> None:
    quarantine = Quarantine(2)
    for index in range(2):
        quarantine.add(envelope(index), reason="poison", actor="root/w0")
    assert quarantine.remaining_capacity == 0
    with pytest.raises(QuarantineFullError):
        quarantine.add(envelope(99), reason="poison", actor="root/w0")
    assert quarantine.overflow_total == 1
    assert len(quarantine) == 2


def test_records_can_be_filtered_and_limited() -> None:
    quarantine = Quarantine(10)
    for index in range(3):
        quarantine.add(envelope(index), reason="poison", actor="root/w0")
    quarantine.add(envelope(9), reason="deadline_exceeded", actor="root/w1")
    assert len(quarantine.records(reason="poison")) == 3
    assert len(quarantine.records(limit=2)) == 2
    assert quarantine.reasons() == {"deadline_exceeded": 1, "poison": 3}
    assert quarantine.get("unknown") is None


def test_take_removes_replayable_records_and_skips_unknown_ids() -> None:
    quarantine = Quarantine(10)
    first, second = envelope(1), envelope(2)
    quarantine.add(first, reason="poison", actor="root/w0")
    quarantine.add(second, reason="poison", actor="root/w0")
    taken = quarantine.take([first.id, "does-not-exist"])
    assert [record.message_id for record in taken] == [first.id]
    assert first.id not in quarantine
    assert second.id in quarantine
    assert quarantine.replayed_total == 1


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    quarantine = Quarantine(10)
    for index in range(3):
        quarantine.add(envelope(index), reason="poison", actor="root/w0")
    path = tmp_path / "quarantine.jsonl"
    assert quarantine.export_jsonl(path) == 3
    loaded = Quarantine.load_jsonl(path)
    assert [record.message_id for record in loaded] == [
        record.message_id for record in quarantine.records()
    ]
    assert all(isinstance(record, QuarantineRecord) for record in loaded)


def test_records_are_immutable_and_closed() -> None:
    quarantine = Quarantine(1)
    record = quarantine.add(envelope(1), reason="poison", actor="root/w0")
    with pytest.raises(ValueError, match="frozen"):
        record.reason = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="extra"):
        QuarantineRecord.model_validate({**record.model_dump(), "surprise": 1})
