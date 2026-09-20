"""JSONL cassettes: ReplayProvider plays them back, RecordingProvider writes them.

Each line stores the canonical request hash and the provider-neutral response. Request bodies are
not stored, so cassettes do not copy prompts or tool outputs (which may hold PII) into the repo.
Recording is meant only for an explicit, budget-capped ``make record`` run with a real key.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_supervision_tree.agentic.budget import BudgetLedger
from agent_supervision_tree.agentic.errors import (
    CassetteFormatError,
    CassetteMissError,
    ProviderError,
)
from agent_supervision_tree.agentic.providers.base import LLMProvider, LLMRequest, LLMResponse

__all__ = ["CassetteEntry", "RecordingMode", "RecordingProvider", "ReplayProvider"]

RecordingMode = Literal["create", "overwrite", "append"]


class CassetteEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: str
    response: LLMResponse


class ReplayProvider:
    """Serve responses by request hash. Identical requests are answered in recorded order."""

    name = "replay"

    def __init__(self, path: Path) -> None:
        self._queues: dict[str, deque[LLMResponse]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = CassetteEntry.model_validate_json(line)
                except ValidationError as exc:
                    raise CassetteFormatError(f"{path.name}:{line_number}: invalid entry") from exc
                self._queues.setdefault(entry.request_sha256, deque()).append(entry.response)

    @property
    def remaining(self) -> int:
        """Responses not yet served (non-zero after a run means the workflow changed)."""
        return sum(len(queue) for queue in self._queues.values())

    def complete(self, request: LLMRequest) -> LLMResponse:
        digest = request.sha256()
        queue = self._queues.get(digest)
        if not queue:
            raise CassetteMissError(digest)
        return queue.popleft()


class RecordingProvider:
    """Wrap a live provider, append every response to a cassette, and enforce a budget.

    ``mode="create"`` (the default) refuses a cassette that already has entries: appending a new
    recording to an old one would make replays serve the stale responses first, because identical
    requests are answered in recorded order. Use ``"overwrite"`` to re-record from scratch or
    ``"append"`` to extend a recording deliberately. Failed calls that were still billed
    (refusals after partial output, truncations) are charged to the budget as well.
    """

    name = "recording"

    def __init__(
        self,
        inner: LLMProvider,
        path: Path,
        *,
        budget: BudgetLedger,
        mode: RecordingMode = "create",
    ) -> None:
        if mode == "create" and path.exists() and path.stat().st_size > 0:
            raise FileExistsError(
                f"cassette {path.name} already has entries; pass mode='overwrite' to re-record "
                "it or mode='append' to extend it"
            )
        if mode == "overwrite":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        self._inner = inner
        self._path = path
        self._budget = budget

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._budget.reserve_step()
        try:
            response = self._inner.complete(request)
        except ProviderError as exc:
            if exc.usage is not None:
                self._budget.record_usage(model=exc.model or "unknown", usage=exc.usage)
            raise
        entry = CassetteEntry(request_sha256=request.sha256(), role=request.role, response=response)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(entry.model_dump_json() + "\n")
        # Charge after persisting so a paid response is never lost when the budget trips.
        self._budget.record_usage(model=response.model, usage=response.usage)
        return response
