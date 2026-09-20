"""Hash-chained audit of restarts, escalations, quarantines, approvals and budget events.

Entries use the exact format of :mod:`agent_supervision_tree.agentic.audit` (``seq``, ``ts``,
``actor``, ``action``, ``data``, ``prev_hash``, ``hash``), so an exported file is verified by
the same ``verify_file`` and the same CLI. The chain is kept in memory for the runtime's hot
path (an ``fsync`` per restart would dominate restart latency) and written through to a JSONL
file with buffered I/O when a path is given; ``flush`` makes the file complete. Data is passed
through the PII redaction before it is hashed, so redaction never breaks the chain.

Every ``anchor_every`` entries (and on ``flush``/``close``) the current head and entry count are
appended to ``<path>.anchor``. A hash chain cannot detect its own truncation -- a shorter prefix
is still a valid chain -- so without a head recorded elsewhere an operator has to remember one
and pass ``--expect-head``. With the anchor file, ``agent-supervision-tree audit verify`` fails
closed on truncation and rollback by default. Shipping the anchor (or the head itself) off the
box is still what protects against someone who can rewrite both files.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import JsonValue

from agent_supervision_tree.agentic.audit import (
    GENESIS_HASH,
    AuditEntry,
    anchor_path_for,
    append_anchor,
    entry_hash,
)
from agent_supervision_tree.core.canonical import canonical_json
from agent_supervision_tree.core.redaction import redact_value

__all__ = ["AuditChain"]


class AuditChain:
    def __init__(
        self,
        path: Path | None = None,
        *,
        max_entries: int = 50_000,
        anchor_every: int = 500,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if anchor_every < 1:
            raise ValueError("anchor_every must be at least 1")
        self._path = path
        self._clock = clock
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self._seq = 0
        self._head = GENESIS_HASH
        self._handle: TextIO | None = None
        self._anchor_every = anchor_every
        self._anchored_at = 0
        self._anchor_path = anchor_path_for(path) if path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8", newline="\n")
            anchor = self._anchor_path
            if anchor is not None and anchor.exists():
                anchor.unlink()  # the log was just truncated: its old anchors describe nothing

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
            if self._seq - self._anchored_at >= self._anchor_every:
                self._anchor()
        return entry

    @property
    def anchor_path(self) -> Path | None:
        """Where the head is published, so truncation fails verification without an operator."""
        return self._anchor_path

    def _anchor(self) -> None:
        """Publish the head next to the log. The log is flushed first, never after."""
        if self._handle is None or self._anchor_path is None or self._seq == self._anchored_at:
            return
        self._handle.flush()
        append_anchor(self._anchor_path, entries=self._seq, head=self._head)
        self._anchored_at = self._seq

    def recent(self, limit: int = 100, *, action: str | None = None) -> list[AuditEntry]:
        items = [e for e in self._entries if action is None or e.action == action]
        return items[-limit:]

    def count(self, action: str) -> int:
        return sum(1 for entry in self._entries if entry.action == action)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._anchor()

    def close(self) -> None:
        if self._handle is not None:
            self._anchor()
            self._handle.close()
            self._handle = None

    def export_jsonl(self, path: Path) -> int:
        """Write the retained entries; a complete chain only if none were evicted."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in self._entries:
                handle.write(canonical_json(entry.model_dump(mode="json")) + "\n")
        return len(self._entries)
