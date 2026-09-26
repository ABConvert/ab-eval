"""Small-sample statistics: percentile bootstrap CIs and paired differences. Pure Python."""

from __future__ import annotations

import random
from statistics import mean


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)


def bootstrap_ci(
    values: list[float], *, iters: int = 2_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean. None when there are fewer than two values.

    2,000 resamples: the interval stops moving in the second decimal well before that, and the
    dashboard computes this for every slice of every run.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(iters))
    alpha = (1 - level) / 2
    return round(means[int(alpha * iters)], 4), round(means[int((1 - alpha) * iters) - 1], 4)


def paired_bootstrap_diff(
    a: list[float], b: list[float], *, iters: int = 2_000, seed: int = 0, level: float = 0.95
) -> tuple[float, tuple[float, float] | None]:
    """Mean of (b - a) over paired cases with a bootstrap CI; positive favours b."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    if not diffs:
        return 0.0, None
    return round(mean(diffs), 4), bootstrap_ci(diffs, iters=iters, seed=seed, level=level)
