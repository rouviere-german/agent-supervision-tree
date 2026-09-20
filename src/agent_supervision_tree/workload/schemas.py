"""Closed schemas for everything that crosses an actor boundary.

Every model forbids extra fields and is validated in strict mode, so text injected into a
document cannot smuggle a ``webhook_url``, an ``approved`` flag or a routing hint through the
model's output: the extraction either matches the schema exactly or the message is rejected.
Where a message goes next is decided by the pipeline wiring, never by message content.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

__all__ = [
    "EXTRACTION_SCHEMA",
    "DocumentMessage",
    "Extraction",
    "FaultCommand",
    "LineItem",
    "validate_extraction",
]


class _Closed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class LineItem(_Closed):
    description: str = Field(min_length=1, max_length=200)
    amount_cents: int = Field(ge=0, le=1_000_000_000)


class Extraction(_Closed):
    """What an extractor worker may produce: schema output only, no routing, no approvals."""

    document_id: str = Field(pattern=r"^doc-[0-9a-f]{6,32}$")
    vendor: str = Field(min_length=1, max_length=120)
    currency: Literal["EUR", "USD"]
    total_cents: int = Field(ge=0, le=1_000_000_000)
    line_items: list[LineItem] = Field(min_length=1, max_length=20)
    due_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    def consistent(self) -> bool:
        return self.total_cents == sum(item.amount_cents for item in self.line_items)


class DocumentMessage(_Closed):
    """The ingress payload: an id and untrusted text."""

    id: str = Field(pattern=r"^doc-[0-9a-f]{6,32}$")
    text: str = Field(min_length=1, max_length=20_000)


class FaultCommand(_Closed):
    """Admin-scope fault injection: crash an actor, hang provider calls, inject poison."""

    op: Literal["crash", "hang", "poison", "fault_rate"]
    path: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_/-]{0,255}$")
    count: int = Field(default=1, ge=1, le=1_000)
    rate: float | None = Field(default=None, ge=0.0, le=1.0)


EXTRACTION_SCHEMA: dict[str, JsonValue] = Extraction.model_json_schema()


def validate_extraction(parsed: object) -> Extraction:
    """Strictly validate model output. Raises ``pydantic.ValidationError`` on any deviation."""
    return Extraction.model_validate(parsed, strict=True)
