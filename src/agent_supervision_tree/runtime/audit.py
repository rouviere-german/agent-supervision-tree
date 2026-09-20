"""Hash-chained audit of restarts, escalations, quarantines, approvals and budget events.

Entries use the exact format of :mod:`agent_supervision_tree.agentic.audit` (``seq``, ``ts``,
``actor``, ``action``, ``data``, ``prev_hash``, ``hash``), so an exported file is verified by
the same ``verify_file`` and the same CLI. The chain is kept in memory for the runtime's hot
path (an ``fsync`` per restart would dominate restart latency) and written through to a JSONL
file with buffered I/O when a path is given; ``flush`` makes the file complete. Data is passed
through the PII redaction before it is hashed, so redaction never breaks the chain.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import JsonValue

from agent_supervision_tree.agentic.audit import GENESIS_HASH, AuditEntry, entry_hash
from agent_supervision_tree.core.canonical import canonical_json
from agent_supervision_tree.core.redaction import redact_value

__all__ = ["AuditChain"]


class AuditChain:
    def __init__(
        self,
        path: Path | None = None,
        *,
        max_entries: int = 50_000,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._path = path
        self._clock = clock
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self._seq = 0
        self._head = GENESIS_HASH
        self._handle: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8", newline="\n")

    @property
    def head(self) -> str:
        return self._head

    @property
    def path(self) -> Path | None:
        return self._path

    def __len__(self) -> int:
        return self._seq

    def append(
        self, actor: str, action: str, data: Mapping[str, JsonValue] | None = None
    ) -> AuditEntry:
        fields: dict[str, Any] = {
            "seq": self._seq,
            "ts": self._clock().isoformat(),
            "actor": actor,
            "action": action,
            "data": redact_value(dict(data or {})),
            "prev_hash": self._head,
        }
        fields["hash"] = entry_hash(fields)
        entry = AuditEntry.model_validate(fields)
        self._entries.append(entry)
        self._seq += 1
        self._head = entry.hash
        if self._handle is not None:
            self._handle.write(canonical_json(fields) + "\n")
        return entry

    def recent(self, limit: int = 100, *, action: str | None = None) -> list[AuditEntry]:
        items = [e for e in self._entries if action is None or e.action == action]
        return items[-limit:]

    def count(self, action: str) -> int:
        return sum(1 for entry in self._entries if entry.action == action)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def export_jsonl(self, path: Path) -> int:
        """Write the retained entries; a complete chain only if none were evicted."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in self._entries:
                handle.write(canonical_json(entry.model_dump(mode="json")) + "\n")
        return len(self._entries)
