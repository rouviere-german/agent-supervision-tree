from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_supervision_tree.agentic.budget import BudgetLedger, BudgetLimits
from agent_supervision_tree.agentic.errors import BudgetError, BudgetExceededError
from agent_supervision_tree.agentic.pricing import cost_usd_micros
from agent_supervision_tree.agentic.providers.base import Usage

pytestmark = pytest.mark.unit


def test_steps_are_checked_before_each_call() -> None:
    ledger = BudgetLedger(BudgetLimits.of(steps=2, tokens=1_000, usd=1.0))
    ledger.reserve_step()
    ledger.reserve_step()
    with pytest.raises(BudgetExceededError) as info:
        ledger.reserve_step()
    assert info.value.dimension == "steps"
    assert ledger.state.steps == 2


def test_token_overrun_is_recorded_and_blocks_further_steps() -> None:
    ledger = BudgetLedger(BudgetLimits.of(steps=10, tokens=100, usd=1.0))
    ledger.reserve_step()
    with pytest.raises(BudgetExceededError, match="tokens"):
        ledger.record_usage(model="fake", usage=Usage(input_tokens=80, output_tokens=40))
    assert ledger.state.tokens == 120
    with pytest.raises(BudgetExceededError, match="tokens"):
        ledger.reserve_step()


def test_usd_is_charged_from_the_price_table() -> None:
    ledger = BudgetLedger(BudgetLimits.of(steps=10, tokens=1_000_000, usd=0.02))
    with pytest.raises(BudgetExceededError, match="usd_micros"):
        ledger.record_usage(
            model="claude-opus-5", usage=Usage(input_tokens=1_000, output_tokens=1_000)
        )
    assert ledger.state.usd_micros == 30_000


def test_state_survives_restart_and_limits_cannot_be_raised(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    limits = BudgetLimits.of(steps=5, tokens=1_000, usd=1.0)
    ledger = BudgetLedger(limits, path)
    ledger.reserve_step()
    ledger.record_usage(model="claude-sonnet-5", usage=Usage(input_tokens=10, output_tokens=5))

    resumed = BudgetLedger(limits, path)
    assert resumed.state == ledger.state
    resumed.reserve_step()
    assert BudgetLedger(limits, path).state.steps == 2

    with pytest.raises(BudgetError, match="cannot be raised"):
        BudgetLedger(BudgetLimits.of(steps=50, tokens=1_000, usd=1.0), path)
    tighter = BudgetLedger(BudgetLimits.of(steps=3, tokens=500, usd=0.5), path)
    assert tighter.state.steps == 2


_operations = st.lists(
    st.one_of(
        st.just(("step", 0, 0)),
        st.just(("reload", 0, 0)),
        st.tuples(st.just("usage"), st.integers(0, 400), st.integers(0, 400)),
    ),
    max_size=30,
)


@settings(max_examples=30)
@given(_operations)
def test_persisted_budget_never_resets_or_exceeds_step_limit(
    operations: list[tuple[str, int, int]],
) -> None:
    limits = BudgetLimits.of(steps=10, tokens=3_000, usd=0.05)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "budget.json"
        ledger = BudgetLedger(limits, path)
        expected_steps = expected_tokens = 0
        for kind, input_tokens, output_tokens in operations:
            if kind == "reload":
                ledger = BudgetLedger(limits, path)
            elif kind == "step":
                try:
                    ledger.reserve_step()
                    expected_steps += 1
                except BudgetExceededError:
                    pass
            else:
                expected_tokens += input_tokens + output_tokens
                with contextlib.suppress(BudgetExceededError):
                    ledger.record_usage(
                        model="claude-sonnet-5",
                        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                    )
            assert ledger.state.steps == expected_steps <= limits.max_steps
            assert ledger.state.tokens == expected_tokens


def test_ledgers_sharing_a_file_cannot_double_spend(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    limits = BudgetLimits.of(steps=2, tokens=1_000, usd=1.0)
    first = BudgetLedger(limits, path)
    second = BudgetLedger(limits, path)
    first.reserve_step()
    second.reserve_step()
    with pytest.raises(BudgetExceededError, match="steps"):
        first.reserve_step()
    with pytest.raises(BudgetExceededError, match="steps"):
        second.reserve_step()
    first.record_usage(model="claude-sonnet-5", usage=Usage(input_tokens=10, output_tokens=0))
    second.record_usage(model="claude-sonnet-5", usage=Usage(input_tokens=5, output_tokens=0))
    assert BudgetLedger(limits, path).state.tokens == 15


def test_a_limit_tightened_by_another_instance_is_never_loosened_again(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    loose = BudgetLedger(BudgetLimits.of(steps=5, tokens=1_000, usd=1.0), path)
    BudgetLedger(BudgetLimits.of(steps=1, tokens=1_000, usd=1.0), path)
    loose.reserve_step()
    with pytest.raises(BudgetExceededError, match="steps"):
        loose.reserve_step()
    assert loose.limits.max_steps == 1


def test_a_deleted_ledger_file_does_not_reset_an_open_ledger(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(BudgetLimits.of(steps=2, tokens=1_000, usd=1.0), path)
    ledger.reserve_step()
    ledger.reserve_step()
    path.unlink()
    with pytest.raises(BudgetExceededError, match="steps"):
        ledger.reserve_step()


def test_pricing() -> None:
    assert cost_usd_micros("fake-model", 10_000, 10_000) == 0
    assert cost_usd_micros("claude-haiku-4-5", 1_000, 1_000) == 6_000
    assert cost_usd_micros("claude-sonnet-5", 1, 1) == 12
    assert cost_usd_micros("some-future-model", 1, 1) == cost_usd_micros("claude-opus-5", 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        cost_usd_micros("claude-opus-5", -1, 0)
