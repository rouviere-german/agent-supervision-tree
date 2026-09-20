"""Hash-chained, append-only audit log (JSONL) with a verify CLI.

Each line is the canonical JSON of ``{seq, ts, actor, action, data, prev_hash, hash}`` where
``hash = sha256(canonical_json(entry without hash))`` and ``prev_hash`` is the previous entry's
hash (64 zeros for the first). Verification detects edited, deleted, inserted and reordered
entries and re-encoded lines. Truncating the *tail* leaves a valid shorter chain, which
re-hashing the log can never catch, so the head is also recorded outside the log:

* ``--expect-head HASH`` compares the final head with one the caller kept, and
* an **anchor file** (``<log>.anchor``) records ``{entries, head, ts}`` as the log grows. Every
  anchor must still match the log's prefix of that length, so truncation, rollback and any
  rewrite behind the last anchor fail verification without an operator having to remember a
  hash. Appending past the last anchor stays legal: that is what an append-only log does.

An anchor is only as good as its own storage -- whoever can rewrite both files can make them
agree again -- which is why the runbook also ships the head off the box (THREAT_MODEL, T1).

Writers take a file lock per append, so several writers (threads, instances or processes) extend
one chain instead of forking it. A writer also refuses to extend a file that shrank or whose last
known entry changed since it last read it, so truncation or rewriting while the log is open is
detected at the next append even without an external head anchor.

Usage: ``python -m agent_supervision_tree.agentic.audit verify audit.jsonl [--expect-head HASH]``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_supervision_tree.agentic.errors import AuditIntegrityError
from agent_supervision_tree.core.canonical import canonical_json, sha256_hex
from agent_supervision_tree.core.fileio import file_lock
from agent_supervision_tree.core.redaction import redact_value

__all__ = [
    "ANCHOR_SUFFIX",
    "GENESIS_HASH",
    "Anchor",
    "AuditEntry",
    "AuditLog",
    "VerifyResult",
    "anchor_path_for",
    "append_anchor",
    "entry_hash",
    "main",
    "read_anchors",
    "verify_file",
]

GENESIS_HASH = "0" * 64
ANCHOR_SUFFIX = ".anchor"


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0)
    ts: str
    actor: str
    action: str
    data: dict[str, JsonValue]
    prev_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def entry_hash(fields: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of an entry without its ``hash`` field."""
    return sha256_hex(canonical_json({k: v for k, v in fields.items() if k != "hash"}))


_entry_hash = entry_hash


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    entries: int
    head: str
    error: str | None = None
    line: int | None = None
    anchors: int = 0


@dataclass(frozen=True, slots=True)
class Anchor:
    """A head published outside the log: it had ``entries`` entries ending in ``head``."""

    entries: int
    head: str
    ts: str = ""


def anchor_path_for(path: Path) -> Path:
    """The conventional anchor file for a log: the log's name plus ``.anchor``."""
    return path.with_name(path.name + ANCHOR_SUFFIX)


def append_anchor(
    path: Path, *, entries: int, head: str, ts: str | None = None, sync: bool = True
) -> Anchor:
    """Append one anchor record. The file is append-only and never rewritten."""
    if entries < 0:
        raise ValueError("entries must be non-negative")
    anchor = Anchor(entries, head, ts or datetime.now(UTC).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {"entries": entries, "head": head, "ts": anchor.ts}
    line = (canonical_json(fields) + "\n").encode("utf-8")
    with file_lock(path), path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        if sync:
            os.fsync(handle.fileno())
    return anchor


def read_anchors(path: Path) -> list[Anchor]:
    """Every anchor in the file, oldest first. A missing file has no anchors."""
    if not path.exists():
        return []
    anchors: list[Anchor] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError("an anchor record must be an object")
        anchors.append(
            Anchor(int(record["entries"]), str(record["head"]), str(record.get("ts", "")))
        )
    return anchors


def _head_after(raw: bytes, entries: int) -> str:
    """The head of an already-verified chain after its first ``entries`` entries."""
    if entries == 0:
        return GENESIS_HASH
    line = raw.split(b"\n")[entries - 1]
    return str(json.loads(line.decode("utf-8"))["hash"])


def _check_anchors(raw: bytes, result: VerifyResult, anchor_path: Path) -> VerifyResult:
    """Every anchor must match the log's prefix of that length."""
    try:
        anchors = read_anchors(anchor_path)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, UnicodeDecodeError):
        return VerifyResult(False, result.entries, result.head, "unreadable anchor file", None)
    for anchor in anchors:
        if anchor.entries > result.entries:
            error = (
                f"audit log is shorter than its anchor ({result.entries} entries, anchored at "
                f"{anchor.entries}: truncated or rolled back)"
            )
            return VerifyResult(False, result.entries, result.head, error, None, len(anchors))
        if _head_after(raw, anchor.entries) != anchor.head:
            error = f"anchor mismatch at entry {anchor.entries} (history was rewritten)"
            return VerifyResult(
                False, result.entries, result.head, error, anchor.entries or None, len(anchors)
            )
    return VerifyResult(True, result.entries, result.head, None, None, len(anchors))


def _verify_chain(raw: bytes, *, prev: str, seq: int) -> VerifyResult:  # noqa: PLR0911
    """Verify ``\\n``-terminated entries that continue a chain whose head is (``seq``, ``prev``).

    Line numbers in the result are 1-based positions in the whole file (entry ``seq`` is on line
    ``seq + 1``). Each early return is a distinct, named failure mode, which keeps reports precise.
    """
    if raw and not raw.endswith(b"\n"):
        count = raw.count(b"\n")
        return VerifyResult(False, seq + count, prev, "truncated final line", seq + count + 1)
    count = seq
    # Split on "\n" only: bytes.splitlines() also splits on "\r", which would let a "\n" -> "\r"
    # substitution pass verification.
    lines = raw[:-1].split(b"\n") if raw else []
    for line in lines:
        line_number = count + 1
        try:
            text = line.decode("utf-8")
            obj = json.loads(text)
            # Strict: a lax validator would read false or 0.0 as seq 0, giving one entry several
            # valid encodings.
            entry = AuditEntry.model_validate(obj, strict=True)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError):
            return VerifyResult(False, count, prev, "malformed entry", line_number)
        if canonical_json(obj) != text:
            return VerifyResult(False, count, prev, "non-canonical encoding", line_number)
        if entry.seq != count:
            return VerifyResult(False, count, prev, "sequence gap or reordering", line_number)
        if entry.prev_hash != prev:
            return VerifyResult(False, count, prev, "broken hash chain", line_number)
        if _entry_hash(obj) != entry.hash:
            return VerifyResult(False, count, prev, "entry content modified", line_number)
        prev = entry.hash
        count += 1
    return VerifyResult(True, count, prev)


def _last_line(raw: bytes) -> bytes:
    """The final ``\\n``-terminated line of ``raw`` (empty for empty input)."""
    if not raw:
        return b""
    return raw[raw.rfind(b"\n", 0, len(raw) - 1) + 1 :]


def verify_file(
    path: Path, *, expect_head: str | None = None, anchor: Path | None = None
) -> VerifyResult:
    """Verify the whole chain, and its anchors when an anchor file is given.

    Returns the first failure with its 1-based line number. ``anchor`` catches what the chain
    cannot catch by itself: a truncated tail, a rollback, or a rewrite behind an anchored head.
    """
    if not path.exists():
        return VerifyResult(False, 0, GENESIS_HASH, "audit log not found", None)
    raw = path.read_bytes()
    result = _verify_chain(raw, prev=GENESIS_HASH, seq=0)
    if not result.ok:
        return result
    if expect_head is not None and expect_head != result.head:
        return VerifyResult(
            False, result.entries, result.head, "head mismatch (truncated or rolled back)", None
        )
    if anchor is not None:
        return _check_anchors(raw, result, anchor)
    return result


class AuditLog:
    """Append-only writer. Opening an existing file verifies it first and refuses a broken one."""

    def __init__(
        self,
        path: Path,
        *,
        redact: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._path = path
        self._redact = redact
        self._clock = clock
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            raw = path.read_bytes() if path.exists() else b""
        result = _verify_chain(raw, prev=GENESIS_HASH, seq=0)
        if not result.ok:
            raise AuditIntegrityError(f"{path.name}: {result.error} (line {result.line})")
        self._seq = result.entries
        self._head = result.head
        self._size = len(raw)
        self._last = _last_line(raw)

    @property
    def head(self) -> str:
        return self._head

    @property
    def path(self) -> Path:
        return self._path

    def _catch_up(self) -> None:
        """Adopt entries other writers appended; refuse a file that shrank or was rewritten.

        Called with the file lock held. Only the last known entry and the bytes after it are read,
        so an append stays cheap however long the log grows.
        """
        size = self._path.stat().st_size if self._path.exists() else 0
        if size < self._size:
            raise AuditIntegrityError(
                f"{self._path.name}: log shrank since it was last read (truncated or rolled back)"
            )
        if size == self._size and not self._last:
            return
        with self._path.open("rb") as handle:
            handle.seek(self._size - len(self._last))
            if handle.read(len(self._last)) != self._last:
                raise AuditIntegrityError(
                    f"{self._path.name}: history was rewritten since it was last read"
                )
            appended = handle.read()
        if not appended:
            return
        result = _verify_chain(appended, prev=self._head, seq=self._seq)
        if not result.ok:
            raise AuditIntegrityError(f"{self._path.name}: {result.error} (line {result.line})")
        self._seq = result.entries
        self._head = result.head
        self._size += len(appended)
        self._last = _last_line(appended)

    def append(
        self, actor: str, action: str, data: Mapping[str, JsonValue] | None = None
    ) -> AuditEntry:
        payload = dict(data or {})
        if self._redact:
            payload = redact_value(payload)
        with self._lock, file_lock(self._path):
            self._catch_up()
            fields: dict[str, Any] = {
                "seq": self._seq,
                "ts": self._clock().isoformat(),
                "actor": actor,
                "action": action,
                "data": payload,
                "prev_hash": self._head,
            }
            fields["hash"] = _entry_hash(fields)
            line = (canonical_json(fields) + "\n").encode("utf-8")
            with self._path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            entry = AuditEntry.model_validate(fields)
            self._seq += 1
            self._head = entry.hash
            self._size += len(line)
            self._last = line
            return entry


def _resolve_anchor(path: Path, anchor: Path | None, *, no_anchor: bool) -> Path | None:
    """Anchoring is on by default: the sibling anchor file is used whenever it exists."""
    if no_anchor:
        return None
    if anchor is not None:
        return anchor
    sibling = anchor_path_for(path)
    return sibling if sibling.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit", description="Hash-chained audit log tools")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="verify an audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None, help="expected head hash")
    verify.add_argument(
        "--anchor", type=Path, default=None, help="anchor file (default: <log>.anchor)"
    )
    verify.add_argument(
        "--no-anchor", action="store_true", help="verify the chain alone, ignoring any anchor file"
    )
    args = parser.parse_args(argv)
    anchor = _resolve_anchor(args.path, args.anchor, no_anchor=args.no_anchor)
    result = verify_file(args.path, expect_head=args.expect_head, anchor=anchor)
    report = {
        "ok": result.ok,
        "entries": result.entries,
        "head": result.head,
        "error": result.error,
        "line": result.line,
        "anchors": result.anchors,
    }
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
