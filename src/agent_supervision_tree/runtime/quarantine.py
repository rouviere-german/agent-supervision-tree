"""In-memory quarantine for poison and undeliverable messages, with export and replay.

A message lands here when its attempts are used up, its deadline passed, its kind is not
accepted by the actor it reached, or a policy deferred it (budget exhausted, provider down).
Records are bounded in number so a poison flood cannot grow memory without limit; when the
bound is hit the record is dropped and counted, and the ingress sheds until space is freed.
Records are JSON so an operator can export them (see docs/RUNBOOK_QUARANTINE.md) and replay
them through the admin API once the cause is fixed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agent_supervision_tree.core.redaction import redact_value
from agent_supervision_tree.runtime.errors import QuarantineFullError
from agent_supervision_tree.runtime.messages import Envelope, json_size

__all__ = ["Quarantine", "QuarantineRecord"]


class QuarantineRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str
    kind: str
    payload: JsonValue
    attempts: int = Field(ge=1)
    reason: str
    actor: str
    quarantined_at: str
    traceparent: str | None = None
    replayable: bool = True


class Quarantine:
    def __init__(self, max_records: int = 10_000) -> None:
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        self.max_records = max_records
        self._records: OrderedDict[str, QuarantineRecord] = OrderedDict()
        self.overflow_total = 0
        self.added_total = 0
        self.replayed_total = 0

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, message_id: str) -> bool:
        return message_id in self._records

    @property
    def remaining_capacity(self) -> int:
        return self.max_records - len(self._records)

    def add(self, envelope: Envelope, *, reason: str, actor: str) -> QuarantineRecord:
        """Record ``envelope``; raises :class:`QuarantineFullError` (after counting) when full."""
        payload: JsonValue = (
            redact_value(envelope.payload) if json_size(envelope.payload) else {"opaque": True}
        )
        record = QuarantineRecord(
            message_id=envelope.id,
            kind=envelope.kind,
            payload=payload,
            attempts=envelope.attempt,
            reason=reason,
            actor=actor,
            quarantined_at=datetime.now(UTC).isoformat(),
            traceparent=envelope.traceparent,
            replayable=json_size(envelope.payload) > 0,
        )
        self.added_total += 1
        if envelope.id in self._records:
            self._records[envelope.id] = record
            return record
        if len(self._records) >= self.max_records:
            self.overflow_total += 1
            raise QuarantineFullError(f"quarantine is full ({self.max_records} records)")
        self._records[envelope.id] = record
        return record

    def get(self, message_id: str) -> QuarantineRecord | None:
        return self._records.get(message_id)

    def records(
        self, *, reason: str | None = None, limit: int | None = None
    ) -> list[QuarantineRecord]:
        items = [r for r in self._records.values() if reason is None or r.reason == reason]
        return items[:limit] if limit is not None else items

    def reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.reason] = counts.get(record.reason, 0) + 1
        return dict(sorted(counts.items()))

    def take(self, message_ids: Iterable[str]) -> list[QuarantineRecord]:
        """Remove and return the records for replay (unknown ids are skipped)."""
        taken: list[QuarantineRecord] = []
        for message_id in message_ids:
            record = self._records.pop(message_id, None)
            if record is not None and record.replayable:
                taken.append(record)
                self.replayed_total += 1
            elif record is not None:
                self._records[message_id] = record  # opaque payloads cannot be replayed
        return taken

    def export_jsonl(self, path: Path) -> int:
        """Write one JSON record per line; returns the number of records written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self._records.values():
                handle.write(record.model_dump_json() + "\n")
        return len(self._records)

    @staticmethod
    def load_jsonl(path: Path) -> list[QuarantineRecord]:
        records: list[QuarantineRecord] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(QuarantineRecord.model_validate_json(line))
        return records
