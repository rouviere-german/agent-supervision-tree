"""Token pricing used for USD budgets.

Prices are USD per million tokens, which is numerically equal to micro-USD per token. Source: the
Claude API reference bundled with Claude Code (checked 2026-09). Update them when prices change;
unknown non-fake models are charged at the highest listed price so budgets err on the safe side.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal

__all__ = ["PRICES_PER_MTOK", "cost_usd_micros"]

PRICES_PER_MTOK: Mapping[str, tuple[Decimal, Decimal]] = {
    "claude-opus-5": (Decimal("5.00"), Decimal("25.00")),
    "claude-opus-4-8": (Decimal("5.00"), Decimal("25.00")),
    "claude-sonnet-5": (Decimal("2.00"), Decimal("10.00")),
    "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
}
_MOST_EXPENSIVE = max(PRICES_PER_MTOK.values(), key=lambda price: price[0] + price[1])


def cost_usd_micros(model: str, input_tokens: int, output_tokens: int) -> int:
    """Cost of a call in micro-USD, rounded up. Models named ``fake*`` are free."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if model.startswith("fake"):
        return 0
    input_price, output_price = PRICES_PER_MTOK.get(model, _MOST_EXPENSIVE)
    return math.ceil(input_tokens * input_price + output_tokens * output_price)
