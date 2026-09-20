"""Human approval gates bound to the canonical hash of the exact tool call.

An approval authorizes one specific call for one specific principal: the tool name plus its
validated, canonicalized arguments, as requested by that principal. Changing any argument changes
the hash, so the approval no longer applies, and no other principal (the approver included) can
consume it. Approvals are single-use, expire, and cannot be granted by the requesting principal.

A persisted store can be shared by several processes (an agent worker and a reviewer tool, for
example): every operation takes a file lock and reloads the file first, so decisions made
elsewhere are seen and an approval is consumed at most once. A record whose stored arguments do
not hash to its ``call_hash`` is refused on load, so the arguments shown to an approver are always
the ones the approval releases.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from agent_supervision_tree.agentic.errors import ApprovalError, ApprovalRequiredError
from agent_supervision_tree.agentic.tools import ValidatedCall, tool_call_hash
from agent_supervision_tree.core.fileio import atomic_write_text, file_lock

__all__ = [
    "ApprovalGate",
    "ApprovalRecord",
    "ApprovalStatus",
    "ApprovalStore",
    "tool_call_hash",
]

_ID_ATTEMPTS = 100


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CONSUMED = "consumed"


class ApprovalRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    call_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool: str
    args: dict[str, JsonValue]
    requested_by: str
    requested_at: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str | None = None
    decided_at: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _arguments_match_call_hash(self) -> ApprovalRecord:
        if tool_call_hash(self.tool, self.args) != self.call_hash:
            raise ValueError("call_hash does not match the stored tool and arguments")
        return self


class _StoreFile(BaseModel):
    version: int = 1
    records: list[ApprovalRecord] = Field(default_factory=list)


class ApprovalStore:
    """Approval records, optionally persisted to a JSON file shared safely between processes."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._path = path
        self._ttl = ttl_seconds
        self._clock = clock
        self._id_factory = id_factory
        self._lock = threading.Lock()
        self._records: dict[str, ApprovalRecord] = {}
        with self._synced():
            pass  # validates an existing file up front

    @contextlib.contextmanager
    def _synced(self) -> Iterator[None]:
        """Serialize access (threads, and processes for a persisted store); reload state first."""
        with self._lock:
            if self._path is None:
                yield
                return
            with file_lock(self._path):
                self._load(self._path)
                yield

    def _load(self, path: Path) -> None:
        if not path.exists():
            self._records = {}
            return
        try:
            data = _StoreFile.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise ApprovalError(f"approval store {path.name} failed validation") from exc
        self._records = {record.id: record for record in data.records}

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = _StoreFile(records=list(self._records.values())).model_dump(mode="json")
        atomic_write_text(self._path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _expired(self, record: ApprovalRecord) -> bool:
        return self._clock() - record.requested_at > self._ttl

    def _get(self, request_id: str) -> ApprovalRecord:
        record = self._records.get(request_id)
        if record is None:
            raise ApprovalError(f"unknown approval request {request_id}")
        return record

    def _new_id(self) -> str:
        for _ in range(_ID_ATTEMPTS):
            candidate = self._id_factory()
            if candidate not in self._records:
                return candidate
        raise ApprovalError("could not generate a unique approval request id")

    def get(self, request_id: str) -> ApprovalRecord:
        with self._synced():
            return self._get(request_id)

    def pending(self) -> list[ApprovalRecord]:
        with self._synced():
            return [
                r
                for r in self._records.values()
                if r.status is ApprovalStatus.PENDING and not self._expired(r)
            ]

    def request(self, call: ValidatedCall, *, requested_by: str) -> ApprovalRecord:
        """Create (or return the existing open) request for this exact call and principal."""
        call.ensure_unchanged()
        call_hash = call.call_hash
        with self._synced():
            for record in self._records.values():
                if (
                    record.call_hash == call_hash
                    and record.requested_by == requested_by
                    and record.status is ApprovalStatus.PENDING
                    and not self._expired(record)
                ):
                    return record
            record = ApprovalRecord(
                id=self._new_id(),
                call_hash=call_hash,
                tool=call.tool.name,
                args=call.canonical_args,
                requested_by=requested_by,
                requested_at=self._clock(),
            )
            self._records[record.id] = record
            self._persist()
            return record

    def decide(
        self, request_id: str, *, approver: str, approve: bool, reason: str | None = None
    ) -> ApprovalRecord:
        with self._synced():
            record = self._get(request_id)
            if record.status is not ApprovalStatus.PENDING:
                raise ApprovalError(f"request {request_id} is {record.status.value}, not pending")
            if self._expired(record):
                raise ApprovalError(f"request {request_id} has expired")
            if approver == record.requested_by:
                raise ApprovalError("separation of duties: requester cannot approve own call")
            decided = record.model_copy(
                update={
                    "status": ApprovalStatus.APPROVED if approve else ApprovalStatus.DENIED,
                    "decided_by": approver,
                    "decided_at": self._clock(),
                    "reason": reason,
                }
            )
            self._records[request_id] = decided
            self._persist()
            return decided

    def consume(self, call_hash: str, *, principal: str) -> ApprovalRecord | None:
        """Atomically use up an unexpired approval of ``call_hash`` requested by ``principal``."""
        with self._synced():
            for record in self._records.values():
                if (
                    record.call_hash == call_hash
                    and record.requested_by == principal
                    and record.status is ApprovalStatus.APPROVED
                    and not self._expired(record)
                ):
                    consumed = record.model_copy(update={"status": ApprovalStatus.CONSUMED})
                    self._records[record.id] = consumed
                    self._persist()
                    return consumed
            return None


class ApprovalGate:
    """Decide whether a validated call may run now, on behalf of one principal."""

    def __init__(self, store: ApprovalStore, *, principal: str) -> None:
        self._store = store
        self._principal = principal

    def authorize(self, call: ValidatedCall) -> str:
        """Return the call hash if the call may run; otherwise raise ApprovalRequiredError."""
        call.ensure_unchanged()
        call_hash = call.call_hash
        if not call.tool.needs_approval:
            return call_hash
        if self._store.consume(call_hash, principal=self._principal) is not None:
            return call_hash
        record = self._store.request(call, requested_by=self._principal)
        raise ApprovalRequiredError(record.id, call_hash, call.tool.name)
