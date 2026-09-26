from __future__ import annotations

from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from eval_harness.metrics.similarity import patch_similarity


def record_of(test_case: LLMTestCase) -> dict[str, Any]:
    meta = test_case.metadata or {}
    rec = meta.get("record")
    if not isinstance(rec, dict):
        raise ValueError("test case metadata must carry the attempt record under 'record'")
    return rec


class _RecordMetric(BaseMetric):  # type: ignore[no-untyped-call]
    """Deterministic metric computed from the attempt record. No LLM involved."""

    name = "record metric"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None
        self.error: str | None = None
        self.async_mode = True
        self.strict_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = None

    def compute(self, rec: dict[str, Any]) -> tuple[float, str]:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        try:
            self.score, self.reason = self.compute(record_of(test_case))
        except Exception as e:
            self.error = str(e)
            self.score, self.reason = 0.0, f"error: {e}"
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return self.score is not None and self.score >= (self.threshold or 0.0)

    @property
    def __name__(self) -> str:
        return self.name


class TestsPassMetric(_RecordMetric):  # type: ignore[no-untyped-call]
    """Fraction of the case's tests passing after the attempt. Success = all pass. Primary."""

    name = "Tests pass"

    def compute(self, rec: dict[str, Any]) -> tuple[float, str]:
        if rec.get("status") != "completed":
            return 0.0, f"attempt status {rec.get('status')}: {rec.get('error') or ''}"[:300]
        after = rec.get("tests_after") or {}
        total = int(after.get("total") or 0)
        if total == 0:
            return 0.0, "no tests ran after the attempt"
        passed = int(after.get("passed") or 0)
        failed = int(after.get("failed") or 0)
        exit_ok = int(after.get("exit_code") or 0) == 0
        score = passed / total if exit_ok or failed else 0.0
        return score, f"{passed}/{total} passed, {failed} failed, exit {after.get('exit_code')}"


class NoRegressionMetric(_RecordMetric):  # type: ignore[no-untyped-call]
    """1 when the affected suites show no failure beyond the case baseline, else 0."""

    name = "No regression"

    def compute(self, rec: dict[str, Any]) -> tuple[float, str]:
        regressions = rec.get("regressions")
        if regressions is None:
            return 0.0, "full suite not run (case tests did not pass)"
        if regressions:
            return 0.0, "regressions: " + "; ".join(str(r) for r in regressions[:10])
        suite = rec.get("full_suite") or {}
        return 1.0, f"full suite {suite.get('passed')}/{suite.get('total')} with no new failures"


class LintCleanMetric(_RecordMetric):  # type: ignore[no-untyped-call]
    """1 when lint on touched files is clean."""

    name = "Lint clean"

    def compute(self, rec: dict[str, Any]) -> tuple[float, str]:
        ok = rec.get("lint_ok")
        if ok is None:
            return 0.0, "lint not run"
        return (
            (1.0, "lint clean")
            if ok
            else (0.0, str(rec.get("lint_output") or "lint failed")[-300:])
        )


class PatchSimilarityMetric(_RecordMetric):  # type: ignore[no-untyped-call]
    """Informational: similarity between the generated and the human patch. Never gates."""

    name = "Patch similarity"

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__(threshold)

    def compute(self, rec: dict[str, Any]) -> tuple[float, str]:
        human = str(rec.get("human_patch") or "")
        generated = str(rec.get("generated_diff") or "")
        if not generated:
            return 0.0, "no generated diff"
        score = patch_similarity(generated, human)
        return score, f"similarity {score:.2f} (informational)"
