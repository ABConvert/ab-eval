"""A case is only a measurement if its own reference patch passes it.

bench-v2 scored eight cases where the team's merged change itself failed: a sandbox missing
an npm package (six correct answers marked wrong), a suite that collected no tests, a test
that reads a file the case builder dropped, a two-runner case that passed 181 of 181. Each
was reported as a model failure. Validation now replays the reference patch and marks such
a case invalid, and a model run on an invalid case is recorded as invalid without calling
the model.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eval_harness.harness import runner, testrun
from eval_harness.harness.record import AttemptRecord, save_record


def _result(**kw: object) -> testrun.TestResult:
    base: dict[str, object] = {
        "total": 25,
        "passed": 25,
        "failed": 0,
        "skipped": 0,
        "exit_code": 0,
        "output": "",
    }
    base.update(kw)
    return testrun.TestResult.model_validate(base)


def test_a_passing_reference_is_a_valid_case() -> None:
    assert runner.reference_failure(_result()) is None


def test_a_suite_that_never_loads_makes_the_case_invalid() -> None:
    """25 of 25 passed, one file never loaded because of a missing package, exit 1."""
    r = _result(exit_code=1, failed_tests=["web/a.test.js::<suite error>"])
    reason = runner.reference_failure(r)
    assert reason is not None and "web/a.test.js" in reason


def test_no_tests_with_the_reference_applied_makes_the_case_invalid() -> None:
    reason = runner.reference_failure(_result(total=0, passed=0, exit_code=0))
    assert reason is not None and "no tests ran" in reason


def test_a_failing_named_test_is_named_in_the_reason() -> None:
    r = _result(passed=24, failed=1, exit_code=1, failed_tests=["web/b.test.js::keeps state"])
    assert "web/b.test.js::keeps state" in (runner.reference_failure(r) or "")


def _validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, error: str) -> None:
    from eval_harness import paths

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    save_record(
        AttemptRecord(
            case_id="DEMO-1",
            run_id=runner.VALIDATE_RUN,
            model="human-patch",
            status=status,
            error=error,
            started_at="2026-01-01T00:00:00Z",
        )
    )


def test_a_case_invalid_at_validation_is_skipped_without_calling_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _validated(tmp_path, monkeypatch, "invalid", "the reference patch does not pass its own tests")

    class Boom:
        name = "boom"

        async def run_agent(self, **kw: object) -> object:
            raise AssertionError("the model must not be called on an invalid case")

    class _Case:
        case_id = "DEMO-1"

    rec = asyncio.run(
        runner.run_case(
            _Case(),
            object(),
            adapter=Boom(),
            caps=runner.Caps(),  # type: ignore[arg-type]
            run_id="r1",
            docker=object(),
        )
    )  # type: ignore[arg-type]
    assert rec.status == "invalid"
    assert "reference patch" in (rec.error or "")
    assert rec.cost_usd == 0


def test_a_case_that_passed_validation_is_not_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _validated(tmp_path, monkeypatch, "completed", "")
    assert runner.invalid_by_validation("DEMO-1") is None


def test_a_case_never_validated_runs_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval_harness import paths

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    assert runner.invalid_by_validation("NEVER-1") is None
