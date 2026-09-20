"""The token estimators: a conservative one for reservations, a modelled one for the fakes."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_supervision_tree.agentic.tokens import (
    BYTES_PER_TOKEN,
    estimate_tokens,
    simulate_usage_tokens,
)

pytestmark = pytest.mark.unit

PAYLOAD = json.dumps(
    {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "invoice #4711"}]}],
        "max_tokens": 512,
        "role": "extractor",
    },
    separators=(",", ":"),
)


def test_the_empty_string_still_costs_a_token() -> None:
    assert estimate_tokens("") == 1
    assert simulate_usage_tokens("") == 1


def test_the_reservation_estimate_is_denser_than_four_characters_per_token() -> None:
    """Punctuation-dense JSON tokenizes denser than prose, so the reservation must err high."""
    assert estimate_tokens(PAYLOAD) >= len(PAYLOAD) / BYTES_PER_TOKEN
    assert estimate_tokens(PAYLOAD) > simulate_usage_tokens(PAYLOAD)


def test_non_ascii_text_is_charged_by_its_utf8_bytes() -> None:
    """A byte-level BPE sees three bytes per CJK character, not one."""
    assert estimate_tokens("中" * 40) > estimate_tokens("a" * 40)


@given(text=st.text(min_size=0, max_size=400))
def test_the_reservation_estimate_is_never_below_the_modelled_usage(text: str) -> None:
    """Whatever the text, the reservation covers the usage the fake providers report."""
    assert estimate_tokens(text) >= simulate_usage_tokens(text)


@given(
    left=st.text(min_size=0, max_size=200),
    right=st.text(min_size=1, max_size=200),
)
def test_more_text_never_costs_fewer_tokens(left: str, right: str) -> None:
    assert estimate_tokens(left + right) >= estimate_tokens(left)


def test_the_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_tokens("hello", bytes_per_token=0)
    with pytest.raises(ValueError, match="positive"):
        simulate_usage_tokens("hello", chars_per_token=0)
