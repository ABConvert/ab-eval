"""`deepeval test run tests/test_deepeval_run.py` entry point; needs EVAL_RUN_ID."""

from __future__ import annotations

import os

import pytest
from deepeval import assert_test

from eval_harness.metrics.deterministic import NoRegressionMetric, TestsPassMetric

RUN_ID = os.environ.get("EVAL_RUN_ID")
pytestmark = pytest.mark.deepeval


def _cases() -> list:  # type: ignore[type-arg]
    if not RUN_ID:
        return []
    from eval_harness.collect.cases import load_case
    from eval_harness.harness.record import AttemptRecord, results_dir
    from eval_harness.metrics.dataset import test_case_from

    out = []
    for path in sorted(results_dir(RUN_ID).glob("DEMO-*.json")):
        rec = AttemptRecord.model_validate_json(path.read_text())
        out.append(test_case_from(rec, load_case(rec.case_id)))
    return out


@pytest.mark.skipif(not RUN_ID, reason="set EVAL_RUN_ID to score a run under pytest")
@pytest.mark.parametrize("test_case", _cases(), ids=lambda tc: tc.name or "?")
def test_attempt(test_case) -> None:  # type: ignore[no-untyped-def]
    assert_test(test_case=test_case, metrics=[TestsPassMetric(), NoRegressionMetric()])
