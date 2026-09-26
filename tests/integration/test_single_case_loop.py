import pytest

from eval_harness.adapters.base import Caps
from eval_harness.adapters.fake import HumanPatchAdapter, NoopAdapter
from eval_harness.collect.cases import load_case
from eval_harness.config import load_repos
from eval_harness.harness.runner import run_many

pytestmark = pytest.mark.docker


async def test_validate_then_noop_fails_and_human_patch_resolves() -> None:
    case = load_case("DEMO-2877")
    repo = load_repos()["demo-app"]
    common = {"repo": repo, "caps": Caps(), "concurrency": 1, "retry_errors": True}

    baseline = (
        await run_many(
            [case], adapter=NoopAdapter(), run_id="itest-validate", validate_only=True, **common
        )
    )[0]
    assert baseline.status == "completed", baseline.error
    assert baseline.tests_before is not None and baseline.tests_before.failed == 3
    assert baseline.full_suite is not None and baseline.full_suite.total > 1000
    case.full_suite_baseline_failures = baseline.full_suite.failed_tests

    noop = (await run_many([case], adapter=NoopAdapter(), run_id="itest-noop", **common))[0]
    assert noop.status == "completed" and not noop.resolved
    assert noop.tests_after is not None and noop.tests_after.failed == 3
    assert noop.files_touched == [] and noop.generated_diff == ""

    human = (
        await run_many(
            [case], adapter=HumanPatchAdapter(case.human_patch), run_id="itest-human", **common
        )
    )[0]
    assert human.status == "completed" and human.resolved, human.error or human.tests_after
    assert human.files_touched == ["web/services/demo/ExampleService.ts"]
    assert human.touched_test_files == []
    assert "FIRST_RETURNING_ORDER_INDEX" in human.generated_diff
    assert human.full_suite is not None and human.full_suite.total > 1000
    assert human.regressions == [], human.regressions
    assert human.lint_ok is True
