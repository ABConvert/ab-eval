from eval_harness.report.stats import bootstrap_ci, paired_bootstrap_diff, percentile


def test_percentile_and_ci() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.5) is None
    lo, hi = bootstrap_ci([1.0] * 10 + [0.0] * 10, iters=2000)  # type: ignore[misc]
    assert 0.25 <= lo <= 0.5 <= hi <= 0.75
    assert bootstrap_ci([1.0]) is None
    assert bootstrap_ci([1.0] * 5, iters=200) == (1.0, 1.0)


def test_paired_diff_detects_direction() -> None:
    a = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    diff, ci = paired_bootstrap_diff(a, b, iters=2000)
    assert diff == 0.5 and ci is not None and ci[0] > 0
    diff, ci = paired_bootstrap_diff(a, a, iters=200)
    assert diff == 0.0 and ci == (0.0, 0.0)
