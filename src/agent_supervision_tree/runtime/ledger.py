"""Hierarchical quota ledger: reserve from every ancestor, then settle.

Nodes mirror the supervision tree (``root``, ``root/extractors``, ``root/extractors/w0`` ...).
Each node has a token and a micro-USD limit. Before a model call, a child *reserves* an upper
bound of what the call may cost; the reservation is checked against the remainder
(``limit - spent - reserved``) of the node **and every ancestor** and applied to all of them
atomically, or refused without touching any. After the call the reservation is *settled* with
the actual usage (or released on failure). Because reservations are upper bounds and are held
until settlement, concurrent callers can never overspend a shared ancestor, however many of
them race: the invariant ``spent + reserved <= limit`` holds at every node at every moment.

The guarantee the ledger itself can make is that it never *authorizes* more than a limit:
``spent + reserved - overshoot <= limit`` at every node, at every moment. A provider that
reports more usage than the request allowed is settled honestly anyway (spend is never hidden),
the excess is counted per node as an overshoot and audited, and because it still raises
``spent`` it immediately shrinks what later calls may reserve -- so a lying provider costs at
most one call's excess before the node refuses everything.

The ledger is synchronous and guarded by a lock, so it is safe from asyncio tasks and from
threads alike; there is no ``await`` inside a reservation.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any

from agent_supervision_tree.runtime.errors import QuotaExceededError

__all__ = ["QuotaLedger", "QuotaNode", "Reservation"]

_ids = itertools.count(1)


@dataclass
class QuotaNode:
    path: str
    parent: str | None
    limit_tokens: int
    limit_usd_micros: int
    reserved_tokens: int = 0
    reserved_usd_micros: int = 0
    spent_tokens: int = 0
    spent_usd_micros: int = 0
    overshoot_tokens: int = 0
    overshoot_usd_micros: int = 0
    reservations_total: int = 0
    refusals_total: int = 0
    overshoot_total: int = 0
    children: list[str] = field(default_factory=list)

    @property
    def remaining_tokens(self) -> int:
        return self.limit_tokens - self.spent_tokens - self.reserved_tokens

    @property
    def remaining_usd_micros(self) -> int:
        return self.limit_usd_micros - self.spent_usd_micros - self.reserved_usd_micros

    @property
    def authorized_tokens(self) -> int:
        """Tokens this node ever approved: spend and holds, minus what providers billed over."""
        return self.spent_tokens + self.reserved_tokens - self.overshoot_tokens

    @property
    def authorized_usd_micros(self) -> int:
        """Micro-USD this node ever approved (see :attr:`authorized_tokens`)."""
        return self.spent_usd_micros + self.reserved_usd_micros - self.overshoot_usd_micros

    @property
    def exhausted(self) -> bool:
        return self.remaining_tokens <= 0 or self.remaining_usd_micros <= 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "limit_tokens": self.limit_tokens,
            "limit_usd_micros": self.limit_usd_micros,
            "reserved_tokens": self.reserved_tokens,
            "reserved_usd_micros": self.reserved_usd_micros,
            "spent_tokens": self.spent_tokens,
            "spent_usd_micros": self.spent_usd_micros,
            "overshoot_tokens": self.overshoot_tokens,
            "overshoot_usd_micros": self.overshoot_usd_micros,
            "remaining_tokens": self.remaining_tokens,
            "remaining_usd_micros": self.remaining_usd_micros,
            "reservations_total": self.reservations_total,
            "refusals_total": self.refusals_total,
            "overshoot_total": self.overshoot_total,
        }


@dataclass(eq=False)
class Reservation:
    id: int
    path: str
    tokens: int
    usd_micros: int
    open: bool = True


class QuotaLedger:
    def __init__(self, root: str, *, tokens: int, usd_micros: int) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, QuotaNode] = {}
        self.root = root
        self._nodes[root] = QuotaNode(root, None, tokens, usd_micros)

    # --- structure -----------------------------------------------------------------------

    def add_node(self, path: str, *, tokens: int, usd_micros: int) -> QuotaNode:
        """Add a node under its parent (the path with the last segment removed)."""
        if tokens < 0 or usd_micros < 0:
            raise ValueError("limits must be non-negative")
        parent, _, _ = path.rpartition("/")
        with self._lock:
            if path in self._nodes:
                raise ValueError(f"node {path} already exists")
            if parent not in self._nodes:
                raise ValueError(f"parent {parent!r} of {path} is not in the ledger")
            node = QuotaNode(path, parent, tokens, usd_micros)
            self._nodes[path] = node
            self._nodes[parent].children.append(path)
            return node

    def node(self, path: str) -> QuotaNode:
        node = self._nodes.get(path)
        if node is None:
            raise KeyError(f"no ledger node {path}")
        return node

    def paths(self) -> list[str]:
        return list(self._nodes)

    def _lineage(self, path: str) -> list[QuotaNode]:
        """The node and its ancestors, leaf first."""
        chain: list[QuotaNode] = []
        current: str | None = path
        while current is not None:
            node = self.node(current)
            chain.append(node)
            current = node.parent
        return chain

    # --- reserve / settle ----------------------------------------------------------------

    def reserve(self, path: str, *, tokens: int, usd_micros: int) -> Reservation:
        """Hold ``tokens`` and ``usd_micros`` on ``path`` and every ancestor, or refuse all."""
        if tokens < 0 or usd_micros < 0:
            raise ValueError("reservation amounts must be non-negative")
        with self._lock:
            lineage = self._lineage(path)
            for node in lineage:
                if node.remaining_tokens < tokens:
                    node.refusals_total += 1
                    raise QuotaExceededError(node.path, "tokens", tokens, node.remaining_tokens)
                if node.remaining_usd_micros < usd_micros:
                    node.refusals_total += 1
                    raise QuotaExceededError(
                        node.path, "usd_micros", usd_micros, node.remaining_usd_micros
                    )
            for node in lineage:
                node.reserved_tokens += tokens
                node.reserved_usd_micros += usd_micros
                node.reservations_total += 1
            return Reservation(next(_ids), path, tokens, usd_micros)

    def settle(self, reservation: Reservation, *, tokens: int, usd_micros: int) -> bool:
        """Convert a reservation into spend. Returns False when the spend overshot the hold.

        The excess of a settlement above its hold is spend the ledger never authorized. It is
        recorded as such per node, so :meth:`check_invariants` can tell a provider that billed
        over the request from a ledger that handed out more than a limit.
        """
        if tokens < 0 or usd_micros < 0:
            raise ValueError("settled amounts must be non-negative")
        with self._lock:
            self._close(reservation)
            excess_tokens = max(0, tokens - reservation.tokens)
            excess_usd = max(0, usd_micros - reservation.usd_micros)
            overshoot = bool(excess_tokens or excess_usd)
            for node in self._lineage(reservation.path):
                node.reserved_tokens -= reservation.tokens
                node.reserved_usd_micros -= reservation.usd_micros
                node.spent_tokens += tokens
                node.spent_usd_micros += usd_micros
                node.overshoot_tokens += excess_tokens
                node.overshoot_usd_micros += excess_usd
                if overshoot:
                    node.overshoot_total += 1
            return not overshoot

    def release(self, reservation: Reservation) -> None:
        """Cancel a reservation that produced no spend."""
        with self._lock:
            self._close(reservation)
            for node in self._lineage(reservation.path):
                node.reserved_tokens -= reservation.tokens
                node.reserved_usd_micros -= reservation.usd_micros

    def _close(self, reservation: Reservation) -> None:
        if not reservation.open:
            raise ValueError(f"reservation {reservation.id} is already settled or released")
        reservation.open = False

    # --- introspection -------------------------------------------------------------------

    def remaining(self, path: str) -> tuple[int, int]:
        node = self.node(path)
        return node.remaining_tokens, node.remaining_usd_micros

    def exhausted(self, path: str) -> bool:
        return self.node(path).exhausted

    def overshoots(self) -> list[str]:
        """Paths where a provider reported more usage than the reservation allowed."""
        with self._lock:
            return [node.path for node in self._nodes.values() if node.overshoot_total]

    def check_invariants(self) -> list[str]:
        """Return every violated invariant (empty when the ledger is consistent).

        Spend above a limit that a provider caused by billing over its reservation is *not* a
        ledger violation: it is reported by :meth:`overshoots`, counted per node and audited by
        the provider actor. What must never happen is the ledger authorizing it.
        """
        problems: list[str] = []
        with self._lock:
            for node in self._nodes.values():
                if node.reserved_tokens < 0 or node.reserved_usd_micros < 0:
                    problems.append(f"{node.path}: negative reservation")
                if node.authorized_tokens > node.limit_tokens:
                    problems.append(f"{node.path}: tokens authorized over limit")
                if node.authorized_usd_micros > node.limit_usd_micros:
                    problems.append(f"{node.path}: usd authorized over limit")
                for child_path in node.children:
                    child = self._nodes[child_path]
                    if child.spent_tokens > node.spent_tokens:
                        problems.append(f"{child_path}: spent more tokens than parent")
        return problems

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._subtree(self.root)

    def _subtree(self, path: str) -> dict[str, Any]:
        node = self._nodes[path]
        data = node.snapshot()
        data["children"] = [self._subtree(child) for child in node.children]
        return data
