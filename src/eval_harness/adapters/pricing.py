from __future__ import annotations

from eval_harness.adapters.base import Usage
from eval_harness.config import Prices


def cost_usd(usage: Usage, prices: Prices | None) -> float:
    """USD cost from token counts and per-million-token rates. No prices → 0.0."""
    if prices is None:
        return 0.0
    per = 1_000_000
    return round(
        usage.input_tokens * prices.input / per
        + usage.output_tokens * prices.output / per
        + usage.cache_read_tokens * prices.cache_read / per
        + usage.cache_write_tokens * prices.cache_write / per,
        6,
    )
