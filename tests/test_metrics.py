from typing import Any

from deepeval.test_case import LLMTestCase

from eval_harness.metrics.deterministic import (
    LintCleanMetric,
    NoRegressionMetric,
    PatchSimilarityMetric,
    TestsPassMetric,
)

DIFF = "diff --git a/web/x.ts b/web/x.ts\n--- a/web/x.ts\n+++ b/web/x.ts\n@@ -1 +1 @@\n-a\n+b\n"


def _tc(**rec: Any) -> LLMTestCase:
    base: dict[str, Any] = {
        "status": "completed",
        "tests_after": {"total": 36, "passed": 36, "failed": 0, "exit_code": 0},
        "regressions": [],
        "full_suite": {"total": 9740, "passed": 9740},
        "lint_ok": True,
        "generated_diff": DIFF,
        "human_patch": DIFF,
    }
    base.update(rec)
    return LLMTestCase(input="task", actual_output="diff", metadata={"record": base})


def test_tests_pass_metric_scores_fraction_and_gates_on_all() -> None:
    m = TestsPassMetric()
    assert m.measure(_tc()) == 1.0 and m.is_successful()
    m = TestsPassMetric()
    assert (
        m.measure(_tc(tests_after={"total": 36, "passed": 33, "failed": 3, "exit_code": 1}))
        == 33 / 36
    )
    assert not m.is_successful()
    m = TestsPassMetric()
    assert m.measure(_tc(status="invalid", error="tests pass before any fix")) == 0.0
    assert "invalid" in (m.reason or "")


def test_no_regression_and_lint() -> None:
    assert NoRegressionMetric().measure(_tc()) == 1.0
    m = NoRegressionMetric()
    assert m.measure(_tc(regressions=["web/a.test.js::x"])) == 0.0 and "web/a.test.js::x" in (
        m.reason or ""
    )
    assert NoRegressionMetric().measure(_tc(regressions=None)) == 0.0
    assert LintCleanMetric().measure(_tc()) == 1.0
    assert LintCleanMetric().measure(_tc(lint_ok=False, lint_output="1 error")) == 0.0
    m = LintCleanMetric()
    assert m.measure(_tc(lint_ok=None)) == 0.0 and m.reason == "lint not run"


def test_patch_similarity_never_gates() -> None:
    m = PatchSimilarityMetric()
    assert m.measure(_tc()) == 1.0 and m.is_successful()
    m = PatchSimilarityMetric()
    assert m.measure(_tc(generated_diff="")) == 0.0 and m.is_successful()
    assert m.__name__ == "Patch similarity"


def test_missing_record_is_an_error_not_an_exception() -> None:
    m = TestsPassMetric()
    assert m.measure(LLMTestCase(input="t", actual_output="o")) == 0.0
    assert m.error is not None and not m.is_successful()


def test_skipping_intervals_keeps_every_point_estimate() -> None:
    """ci=False must change only the intervals — the ranking reads the rest."""
    import json
    from pathlib import Path

    from eval_harness.report.render import run_metrics
    from eval_harness.report.summary import RunSummary

    cases = json.loads((Path(__file__).parent / "fixtures" / "summary-two-cases.json").read_text())
    s = RunSummary.model_validate(cases)
    with_ci = run_metrics(s, ci=True)
    without = run_metrics(s, ci=False)

    assert with_ci["overall"]["rate"] == without["overall"]["rate"]
    assert with_ci["overall"]["resolved"] == without["overall"]["resolved"]
    assert with_ci["cost_per_solved_case"] == without["cost_per_solved_case"]
    assert with_ci["overall"]["ci95"] is not None
    assert without["overall"]["ci95"] is None


def test_metrics_are_memoised_per_scored_run() -> None:
    """The dashboard asks three pages for the same run's metrics."""
    import json
    from pathlib import Path

    from eval_harness.report import render
    from eval_harness.report.summary import RunSummary

    raw = json.loads((Path(__file__).parent / "fixtures" / "summary-two-cases.json").read_text())
    s = RunSummary.model_validate(raw)
    render._METRICS_CACHE.clear()
    first = render.run_metrics(s)
    assert render.run_metrics(s) is first, "same run, same object"
    s.scored_at = "2099-01-01T00:00:00Z"
    assert render.run_metrics(s) is not first, "re-scored means recompute"
