from pathlib import Path

import pytest

from eval_harness import paths
from eval_harness.adapters.base import Caps
from eval_harness.collect.cases import Case
from eval_harness.harness import record as rec_mod
from eval_harness.harness import runner as runner_mod
from eval_harness.harness.record import AttemptRecord


def _case(cid: str) -> Case:
    return Case(
        case_id=cid,
        repo="r",
        kind="bug_fix",
        base_commit="a" * 40,
        task_prompt="t",
        test_command="x",
        test_files_added=[],
        test_files=["web/a.test.ts"],
        code_files=["web/a.ts"],
        human_patch="",
        human_test_patch="",
        files_changed=2,
        lines_changed=3,
        human_cycle_time_hours=1.0,
        review_comment_count=0,
        merged_at="2026-09-02T00:00:00Z",
        pr_number=1,
        merge_commit="b" * 40,
        test_runner="unit",
        files_dropped=[],
        tier="A",
        kind_source="review",
        author="x",
        author_kind="human",
        spec_level_heuristic="symptom",
        ticket_created_at="2026-09-01T00:00:00Z",
        ticket_started_at=None,
        human_cycle_time_source="created_at",
    )


async def test_run_many_skips_completed_and_retries_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_repo: object
) -> None:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    monkeypatch.setattr(runner_mod, "ensure_image", lambda docker, repo: None)
    monkeypatch.setattr(runner_mod, "Docker", lambda: object())
    rec_mod.save_record(AttemptRecord(case_id="C1", run_id="r", model="noop", status="completed"))
    rec_mod.save_record(AttemptRecord(case_id="C2", run_id="r", model="noop", status="error"))
    ran: list[str] = []

    async def fake_run_case(case: Case, repo: object, **kw: object) -> AttemptRecord:
        ran.append(case.case_id)
        return AttemptRecord(case_id=case.case_id, run_id="r", model="noop", status="completed")

    monkeypatch.setattr(runner_mod, "run_case", fake_run_case)
    repo = demo_repo
    await runner_mod.run_many(
        [_case("C1"), _case("C2"), _case("C3")],
        repo,
        adapter=None,
        caps=Caps(),
        run_id="r",
        concurrency=2,  # type: ignore[arg-type]
    )
    assert ran == ["C3"]
    out = await runner_mod.run_many(
        [_case("C2")],
        repo,
        adapter=None,
        caps=Caps(),
        run_id="r",
        concurrency=1,
        retry_errors=True,  # type: ignore[arg-type]
    )
    assert ran == ["C3", "C2"] and out[0].status == "completed"


def test_regressions_vs_baseline_ignores_baseline_and_own_tests() -> None:
    case = _case("C9")
    case.full_suite_baseline_failures = ["web/x.test.ts::flaky"]
    failed = ["web/x.test.ts::flaky", "web/a.test.ts::own", "web/y.test.ts::new failure"]
    assert runner_mod.regressions_vs_baseline(case, failed) == ["web/y.test.ts::new failure"]
    assert runner_mod.regressions_vs_baseline(case, []) == []


def test_confirm_regressions_needs_a_majority_of_rechecks() -> None:
    flake, real = "web/a.test.js::timing flake", "web/b.test.js::real break"
    cands = [flake, real]
    # fails every time -> a regression; fails once in three -> a flake
    assert runner_mod.confirm_regressions(cands, [[real], [flake, real], [real]], needed=2) == [
        real
    ]
    # a single re-check keeps the old one-shot behaviour
    assert runner_mod.confirm_regressions(cands, [[real]]) == [real]
    assert runner_mod.confirm_regressions(cands, [[], [], []]) == []
    assert runner_mod.confirm_regressions(cands, []) == []


def test_a_criterion_grades_named_tests_not_whole_files() -> None:
    """An imported case passes when its named tests pass, even if the file has other failures."""
    from eval_harness.harness.record import AttemptRecord
    from eval_harness.harness.testrun import TestResult

    after = TestResult(
        total=5,
        passed=4,
        failed=1,
        skipped=0,
        exit_code=1,
        output="",
        failed_tests=["t.py::test_unrelated"],
    )
    base = dict(
        case_id="X",
        run_id="r",
        model="m",
        phase="done",
        started_at="",
        status="completed",
        tests_after=after,
    )
    assert AttemptRecord(**base).resolved is False, "whole-file grading fails on any failure"
    assert (
        AttemptRecord(
            **base, criterion={"fail_to_pass": ["t.py::test_target"], "pass_to_pass": []}
        ).resolved
        is True
    )
    assert (
        AttemptRecord(
            **base, criterion={"fail_to_pass": ["t.py::test_unrelated"], "pass_to_pass": []}
        ).resolved
        is False
    ), "the named test did fail"


def test_a_case_without_a_criterion_is_graded_the_original_way() -> None:
    from eval_harness.harness.record import AttemptRecord
    from eval_harness.harness.testrun import TestResult

    clean = TestResult(total=3, passed=3, failed=0, skipped=0, exit_code=0, output="")
    rec = AttemptRecord(
        case_id="X",
        run_id="r",
        model="m",
        phase="done",
        started_at="",
        status="completed",
        tests_after=clean,
    )
    assert rec.criterion is None
    assert rec.resolved is True
