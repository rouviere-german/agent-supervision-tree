"""Synthetic documents: determinism, injections, poison and the invoice parser."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_supervision_tree.workload.documents import (
    INJECTION_MARKER,
    POISON_MARKER,
    generate_documents,
    parse_invoice,
    render_invoice,
)
from agent_supervision_tree.workload.schemas import Extraction, validate_extraction

pytestmark = pytest.mark.unit


def test_generation_is_deterministic_for_a_seed() -> None:
    first = generate_documents(10, seed=3, injection_rate=0.2)
    second = generate_documents(10, seed=3, injection_rate=0.2)
    assert [doc.model_dump() for doc in first] == [doc.model_dump() for doc in second]
    assert len({doc.id for doc in first}) == 10
    assert generate_documents(3, seed=4)[0].id != first[0].id


def test_rates_are_validated() -> None:
    with pytest.raises(ValueError, match="rates within"):
        generate_documents(1, injection_rate=1.5)
    with pytest.raises(ValueError, match="rates within"):
        generate_documents(-1)


def test_injections_and_poison_are_marked_and_countable() -> None:
    documents = generate_documents(200, seed=1, injection_rate=0.25, poison_rate=0.25)
    injected = [doc for doc in documents if doc.injected]
    poisoned = [doc for doc in documents if doc.poison]
    assert 30 < len(injected) < 70
    assert 30 < len(poisoned) < 70
    assert not set(injected) & set(poisoned)
    assert all(INJECTION_MARKER in doc.text for doc in injected)
    assert all(POISON_MARKER in doc.text for doc in poisoned)


def test_a_clean_document_parses_into_a_consistent_extraction() -> None:
    document = generate_documents(1, seed=9)[0]
    extraction = parse_invoice(document.text)
    assert extraction is not None
    assert extraction.document_id == document.id
    assert extraction.consistent()
    assert validate_extraction(extraction.model_dump(mode="json")) == extraction


def test_an_injected_document_still_parses_to_the_real_invoice_fields() -> None:
    documents = generate_documents(40, seed=2, injection_rate=1.0)
    for document in documents:
        extraction = parse_invoice(document.text)
        assert extraction is not None
        # The injected instructions are just text: they add no fields and change no values.
        assert "evil.example" not in extraction.model_dump_json()


def test_text_that_is_not_an_invoice_does_not_parse() -> None:
    assert parse_invoice("just some prose") is None
    assert parse_invoice("INVOICE doc-abcdef\nVendor: acme\n") is None


@given(
    vendor=st.text(
        alphabet=st.characters(min_codepoint=65, max_codepoint=122), min_size=1, max_size=30
    ),
    currency=st.sampled_from(["EUR", "USD"]),
    items=st.lists(
        st.tuples(
            st.text(
                alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20
            ),
            st.integers(min_value=1, max_value=500_000),
        ),
        min_size=1,
        max_size=8,
    ),
    day=st.integers(min_value=1, max_value=28),
)
def test_rendering_and_parsing_an_invoice_round_trips(
    vendor: str, currency: str, items: list[tuple[str, int]], day: int
) -> None:
    """Whatever invoice is rendered, the parser recovers exactly the fields that went in."""
    document_id = "doc-abc123"
    due = f"2026-11-{day:02d}"
    text = render_invoice(document_id, vendor, currency, items, due)
    extraction = parse_invoice(text)
    assert extraction is not None
    assert extraction.vendor == vendor
    assert extraction.currency == currency
    assert extraction.due_date == due
    assert extraction.total_cents == sum(amount for _, amount in items)
    assert [item.amount_cents for item in extraction.line_items] == [a for _, a in items]
    assert extraction.consistent()


def test_the_extraction_schema_is_closed_and_strict() -> None:
    valid = {
        "document_id": "doc-abcdef",
        "vendor": "acme",
        "currency": "EUR",
        "total_cents": 100,
        "line_items": [{"description": "item", "amount_cents": 100}],
        "due_date": "2026-11-01",
    }
    assert validate_extraction(valid).consistent()
    with pytest.raises(ValueError, match="extra"):
        validate_extraction({**valid, "webhook_url": "https://evil.example"})
    with pytest.raises(ValueError, match="currency"):
        validate_extraction({**valid, "currency": "GBP"})
    with pytest.raises(ValueError, match="document_id"):
        validate_extraction({**valid, "document_id": "../../etc/passwd"})
    with pytest.raises(ValueError, match="line_items"):
        validate_extraction({**valid, "line_items": []})
    assert not Extraction.model_validate({**valid, "total_cents": 999}).consistent()
