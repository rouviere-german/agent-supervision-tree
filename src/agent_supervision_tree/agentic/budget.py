"""Persisted step, token and USD budgets.

State and limits are written atomically after every change, so resuming a run (or replaying it)
continues from the recorded totals instead of starting fresh. Limits may be tightened on resume
but never raised.

Ledgers that share a file (several workers, or a recorder next to an agent loop) take a file lock
and reload the file before every change, so they draw from one budget instead of each spending the
full amount. Counters never move backwards (a deleted or rolled-back file does not reset an open
ledger) and the tightest limits seen by any instance win.

Steps are checked before a call; tokens and cost can only be known afterwards, so a single call
may overshoot, after which every further step is refused.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_supervision_tree.agentic.errors import BudgetError, BudgetExceededError
from agent_supervision_tree.agentic.pricing import cost_usd_micros
from agent_supervision_tree.agentic.providers.base import Usage
from agent_supervision_tree.core.fileio import atomic_write_text, file_lock

__all__ = ["BudgetLedger", "BudgetLimits", "BudgetState"]


class BudgetLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    max_usd_micros: int = Field(ge=0)

    @classmethod
    def of(cls, *, steps: int, tokens: int, usd: float) -> BudgetLimits:
        return cls(max_steps=steps, max_tokens=tokens, max_usd_micros=round(usd * 1_000_000))

    def not_looser_than(self, other: BudgetLimits) -> bool:
        return (
            self.max_steps <= other.max_steps
            and self.max_tokens <= other.max_tokens
            and self.max_usd_micros <= other.max_usd_micros
        )

    def tightest(self, other: BudgetLimits) -> BudgetLimits:
        return BudgetLimits(
            max_steps=min(self.max_steps, other.max_steps),
            max_tokens=min(self.max_tokens, other.max_tokens),
            max_usd_micros=min(self.max_usd_micros, other.max_usd_micros),
        )


class BudgetState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd_micros: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def latest(self, other: BudgetState) -> BudgetState:
        """Element-wise maximum: counters only grow, so the larger value is the newer one."""
        return BudgetState(
            steps=max(self.steps, other.steps),
            input_tokens=max(self.input_tokens, other.input_tokens),
            output_tokens=max(self.output_tokens, other.output_tokens),
            usd_micros=max(self.usd_micros, other.usd_micros),
        )


class _LedgerFile(BaseModel):
    version: int = 1
    limits: BudgetLimits
    state: BudgetState


class BudgetLedger:
    def __init__(self, limits: BudgetLimits, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._limits = limits
        self._state = BudgetState()
        with self._lock, self._file_guard():
            stored = self._read()
            if stored is not None:
                if not limits.not_looser_than(stored.limits):
                    raise BudgetError("budget limits cannot be raised when resuming a run")
                self._state = stored.state
            self._persist()

    def _file_guard(self) -> contextlib.AbstractContextManager[None]:
        return file_lock(self._path) if self._path is not None else contextlib.nullcontext()

    def _read(self) -> _LedgerFile | None:
        if self._path is None or not self._path.exists():
            return None
        try:
            return _LedgerFile.model_validate_json(self._path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise BudgetError(f"budget ledger {self._path.name} failed validation") from exc

    @contextlib.contextmanager
    def _synced(self) -> Iterator[None]:
        """Lock (thread and file) and merge in whatever other ledger instances recorded."""
        with self._lock, self._file_guard():
            stored = self._read()
            if stored is not None:
                self._limits = self._limits.tightest(stored.limits)
                self._state = self._state.latest(stored.state)
            yield

    @property
    def limits(self) -> BudgetLimits:
        with self._synced():
            return self._limits

    @property
    def state(self) -> BudgetState:
        with self._synced():
            return self._state

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = _LedgerFile(limits=self._limits, state=self._state).model_dump(mode="json")
        atomic_write_text(self._path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _ensure_not_exhausted(self) -> None:
        if self._state.tokens > self._limits.max_tokens:
            raise BudgetExceededError("tokens", self._limits.max_tokens, self._state.tokens)
        if self._state.usd_micros > self._limits.max_usd_micros:
            raise BudgetExceededError(
                "usd_micros", self._limits.max_usd_micros, self._state.usd_micros
            )

    def reserve_step(self) -> None:
        """Count one step before making a call. Raises if steps, tokens or cost are exhausted."""
        with self._synced():
            self._ensure_not_exhausted()
            attempted = self._state.steps + 1
            if attempted > self._limits.max_steps:
                raise BudgetExceededError("steps", self._limits.max_steps, attempted)
            self._state = self._state.model_copy(update={"steps": attempted})
            self._persist()

    def record_usage(self, *, model: str, usage: Usage) -> None:
        """Charge a completed call. Persists first, then raises if a limit is now exceeded."""
        with self._synced():
            cost = cost_usd_micros(model, usage.input_tokens, usage.output_tokens)
            self._state = self._state.model_copy(
                update={
                    "input_tokens": self._state.input_tokens + usage.input_tokens,
                    "output_tokens": self._state.output_tokens + usage.output_tokens,
                    "usd_micros": self._state.usd_micros + cost,
                }
            )
            self._persist()
            self._ensure_not_exhausted()
