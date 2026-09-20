"""The hierarchical quota ledger: atomic reservations, settlement and the spend invariant."""

from __future__ import annotations

from contextlib import suppress

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from agent_supervision_tree.runtime.errors import QuotaExceededError
from agent_supervision_tree.runtime.ledger import QuotaLedger, Reservation

pytestmark = pytest.mark.unit


def build_ledger(tokens: int = 1_000, usd: int = 1_000) -> QuotaLedger:
    ledger = QuotaLedger("root", tokens=tokens, usd_micros=usd)
    ledger.add_node("root/pool", tokens=tokens, usd_micros=usd)
    ledger.add_node("root/pool/w0", tokens=tokens // 2, usd_micros=usd // 2)
    ledger.add_node("root/pool/w1", tokens=tokens // 2, usd_micros=usd // 2)
    return ledger


def test_nodes_need_an_existing_parent_and_unique_paths() -> None:
    ledger = build_ledger()
    with pytest.raises(ValueError, match="already exists"):
        ledger.add_node("root/pool", tokens=1, usd_micros=1)
    with pytest.raises(ValueError, match="is not in the ledger"):
        ledger.add_node("root/missing/child", tokens=1, usd_micros=1)
    with pytest.raises(ValueError, match="non-negative"):
        ledger.add_node("root/bad", tokens=-1, usd_micros=1)
    with pytest.raises(KeyError):
        ledger.node("root/nope")


def test_a_reservation_is_held_on_the_node_and_on_every_ancestor() -> None:
    ledger = build_ledger()
    reservation = ledger.reserve("root/pool/w0", tokens=100, usd_micros=40)
    for path in ("root", "root/pool", "root/pool/w0"):
        node = ledger.node(path)
        assert node.reserved_tokens == 100
        assert node.reserved_usd_micros == 40
    assert ledger.node("root/pool/w1").reserved_tokens == 0
    assert ledger.remaining("root") == (900, 960)
    assert ledger.settle(reservation, tokens=80, usd_micros=30)
    assert ledger.node("root").spent_tokens == 80
    assert ledger.node("root").reserved_tokens == 0
    assert ledger.check_invariants() == []


def test_a_refused_reservation_changes_nothing_anywhere() -> None:
    """Siblings with the root's own limit: only the shared ancestor can refuse them."""
    ledger = QuotaLedger("root", tokens=100, usd_micros=100)
    ledger.add_node("root/a", tokens=100, usd_micros=100)
    ledger.add_node("root/b", tokens=100, usd_micros=100)
    held = ledger.reserve("root/a", tokens=70, usd_micros=10)
    with pytest.raises(QuotaExceededError) as info:
        ledger.reserve("root/b", tokens=80, usd_micros=10)
    assert info.value.path == "root"
    assert info.value.dimension == "tokens"
    assert (info.value.requested, info.value.remaining) == (80, 30)
    assert ledger.node("root/b").reserved_tokens == 0
    assert ledger.node("root").reserved_tokens == 70
    assert ledger.node("root").refusals_total == 1
    ledger.release(held)
    assert ledger.node("root").reserved_tokens == 0


def test_the_tightest_limit_in_the_lineage_is_the_one_that_refuses() -> None:
    ledger = build_ledger(tokens=1_000, usd=1_000)
    with pytest.raises(QuotaExceededError) as info:
        ledger.reserve("root/pool/w0", tokens=600, usd_micros=1)
    assert info.value.path == "root/pool/w0"  # the leaf has half the root's limit
    with pytest.raises(QuotaExceededError) as usd_info:
        ledger.reserve("root/pool/w0", tokens=1, usd_micros=600)
    assert usd_info.value.dimension == "usd_micros"


def test_concurrent_reservations_can_never_overspend_a_shared_ancestor() -> None:
    ledger = QuotaLedger("root", tokens=100, usd_micros=100)
    ledger.add_node("root/a", tokens=100, usd_micros=100)
    ledger.add_node("root/b", tokens=100, usd_micros=100)
    first = ledger.reserve("root/a", tokens=60, usd_micros=60)
    with pytest.raises(QuotaExceededError):
        ledger.reserve("root/b", tokens=60, usd_micros=60)
    ledger.settle(first, tokens=60, usd_micros=60)
    second = ledger.reserve("root/b", tokens=40, usd_micros=40)
    ledger.settle(second, tokens=40, usd_micros=40)
    assert ledger.node("root").spent_tokens == 100
    assert ledger.exhausted("root")
    assert ledger.check_invariants() == []


def test_settling_above_the_hold_is_recorded_honestly_and_counted() -> None:
    ledger = build_ledger()
    reservation = ledger.reserve("root/pool/w0", tokens=10, usd_micros=10)
    assert ledger.settle(reservation, tokens=25, usd_micros=10) is False
    assert ledger.node("root").spent_tokens == 25
    assert ledger.node("root").overshoot_total == 1
    assert ledger.node("root/pool/w0").overshoot_total == 1


def test_an_overshoot_is_accounted_separately_and_shrinks_the_remainder() -> None:
    """Spend above the hold is real spend: it is counted, and later calls have less to reserve."""
    ledger = build_ledger(tokens=100, usd=100)
    reservation = ledger.reserve("root/pool/w0", tokens=10, usd_micros=10)
    assert ledger.settle(reservation, tokens=30, usd_micros=40) is False
    root = ledger.node("root")
    assert (root.spent_tokens, root.spent_usd_micros) == (30, 40)
    assert (root.overshoot_tokens, root.overshoot_usd_micros) == (20, 30)
    assert (root.remaining_tokens, root.remaining_usd_micros) == (70, 60)
    # The ledger only ever authorized the 10 it held, which is the invariant it can guarantee.
    assert (root.authorized_tokens, root.authorized_usd_micros) == (10, 10)
    assert ledger.check_invariants() == []
    assert ledger.overshoots() == ["root", "root/pool", "root/pool/w0"]


def test_a_provider_overshoot_past_the_limit_exhausts_the_node() -> None:
    """Once reported usage passes the limit the node is exhausted: the cost breaker trips."""
    ledger = build_ledger(tokens=100, usd=100)
    reservation = ledger.reserve("root/pool/w0", tokens=10, usd_micros=10)
    ledger.settle(reservation, tokens=140, usd_micros=140)
    assert ledger.exhausted("root")
    assert ledger.check_invariants() == []
    with pytest.raises(QuotaExceededError):
        ledger.reserve("root/pool/w0", tokens=1, usd_micros=1)


def test_a_reservation_can_only_be_closed_once() -> None:
    ledger = build_ledger()
    reservation = ledger.reserve("root/pool/w0", tokens=10, usd_micros=10)
    ledger.settle(reservation, tokens=1, usd_micros=1)
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(reservation, tokens=1, usd_micros=1)
    with pytest.raises(ValueError, match="already settled"):
        ledger.release(reservation)


def test_negative_amounts_are_refused() -> None:
    ledger = build_ledger()
    with pytest.raises(ValueError, match="non-negative"):
        ledger.reserve("root", tokens=-1, usd_micros=0)
    reservation = ledger.reserve("root", tokens=1, usd_micros=1)
    with pytest.raises(ValueError, match="non-negative"):
        ledger.settle(reservation, tokens=-1, usd_micros=0)


def test_snapshot_mirrors_the_tree() -> None:
    ledger = build_ledger()
    snapshot = ledger.snapshot()
    assert snapshot["path"] == "root"
    pool = snapshot["children"][0]
    assert pool["path"] == "root/pool"
    assert [child["path"] for child in pool["children"]] == ["root/pool/w0", "root/pool/w1"]
    assert sorted(ledger.paths()) == ["root", "root/pool", "root/pool/w0", "root/pool/w1"]


def test_check_invariants_reports_a_ledger_corrupted_by_hand() -> None:
    ledger = build_ledger()
    ledger.node("root/pool/w0").reserved_tokens = -5
    ledger.node("root").spent_tokens = 10_000
    problems = ledger.check_invariants()
    assert any("negative reservation" in problem for problem in problems)
    assert any("tokens authorized over limit" in problem for problem in problems)


@settings(max_examples=40, deadline=None)
@given(
    amounts=st.lists(st.integers(min_value=0, max_value=40), min_size=1, max_size=20),
    leaf=st.sampled_from(["root/pool/w0", "root/pool/w1"]),
)
def test_spend_never_exceeds_any_limit_for_any_amount_sequence(
    amounts: list[int], leaf: str
) -> None:
    ledger = build_ledger(tokens=200, usd=200)
    open_reservations: list[Reservation] = []
    for amount in amounts:
        with suppress(QuotaExceededError):
            open_reservations.append(ledger.reserve(leaf, tokens=amount, usd_micros=amount))
        assert ledger.check_invariants() == []
    for index, reservation in enumerate(open_reservations):
        if index % 2:
            ledger.release(reservation)
        else:
            ledger.settle(reservation, tokens=reservation.tokens, usd_micros=reservation.usd_micros)
        assert ledger.check_invariants() == []
    root = ledger.node("root")
    assert root.authorized_tokens <= root.limit_tokens
    assert root.overshoot_tokens == 0  # every settlement matched its hold exactly


def _reserve_or_none(ledger: QuotaLedger, path: str, amount: int) -> Reservation | None:
    """Reserve, or return None when the ledger refuses (a refusal is a legal outcome here)."""
    with suppress(QuotaExceededError):
        return ledger.reserve(path, tokens=amount, usd_micros=amount)
    return None


class LedgerMachine(RuleBasedStateMachine):
    """Reserve, settle and release at random; no node may ever exceed its limit."""

    def __init__(self) -> None:
        super().__init__()
        self.ledger = build_ledger(tokens=150, usd=150)
        self.open: list[Reservation] = []
        self.settled_tokens = 0
        self.overshoot_tokens = 0

    @rule(
        leaf=st.sampled_from(["root", "root/pool", "root/pool/w0", "root/pool/w1"]),
        amount=st.integers(min_value=0, max_value=60),
    )
    def reserve(self, leaf: str, amount: int) -> None:
        refused = _reserve_or_none(self.ledger, leaf, amount)
        if refused is not None:
            self.open.append(refused)

    @precondition(lambda self: bool(self.open))
    @rule(index=st.integers(min_value=0), spend=st.integers(min_value=0, max_value=90))
    def settle(self, index: int, spend: int) -> None:
        """Settle with any reported usage, including more than was held (a lying provider)."""
        reservation = self.open.pop(index % len(self.open))
        self.ledger.settle(reservation, tokens=spend, usd_micros=spend)
        self.settled_tokens += spend
        self.overshoot_tokens += max(0, spend - reservation.tokens)

    @precondition(lambda self: bool(self.open))
    @rule(index=st.integers(min_value=0))
    def release(self, index: int) -> None:
        self.ledger.release(self.open.pop(index % len(self.open)))

    @invariant()
    def limits_hold(self) -> None:
        """The ledger never authorizes more than a limit, and it hides no reported spend."""
        assert self.ledger.check_invariants() == []
        for path in self.ledger.paths():
            node = self.ledger.node(path)
            assert node.authorized_tokens <= node.limit_tokens
            assert node.authorized_usd_micros <= node.limit_usd_micros
        root = self.ledger.node("root")
        assert root.spent_tokens == self.settled_tokens
        assert root.overshoot_tokens == self.overshoot_tokens


def test_no_node_ever_exceeds_its_limit_under_random_reservations() -> None:
    """Randomized reserve/settle/release sequences keep spent + reserved <= limit everywhere."""
    run_state_machine_as_test(LedgerMachine, settings=settings(max_examples=40, deadline=None))
