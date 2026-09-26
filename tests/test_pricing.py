from eval_harness.adapters.base import Usage
from eval_harness.adapters.pricing import cost_usd
from eval_harness.config import Prices


def test_cost_uses_all_four_rates() -> None:
    prices = Prices(input=5, output=25, cache_read=0.5, cache_write=6.25)
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=400_000,
    )
    assert cost_usd(usage, prices) == 5 + 2.5 + 1.0 + 2.5


def test_cost_without_prices_is_zero() -> None:
    assert cost_usd(Usage(input_tokens=10), None) == 0.0
