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


def test_an_image_built_from_an_older_dockerfile_is_rebuilt() -> None:
    """Building only when the tag is missing kept a pre-upgrade toolchain forever."""
    from pathlib import Path

    from eval_harness import PROJECT_ROOT
    from eval_harness.config import load_repos
    from eval_harness.harness.runner import DOCKERFILE_LABEL, dockerfile_digest, ensure_image

    repo = next(iter(load_repos(Path(__file__).parent / "fixtures/config/repos.yaml").values()))
    current = dockerfile_digest(PROJECT_ROOT / repo.dockerfile)

    class Docker:
        def __init__(self, label: str | None) -> None:
            self.label = label
            self.built: list[dict[str, str]] = []

        def daemon_ok(self) -> bool:
            return True

        def image_label(self, tag: str, key: str) -> str | None:
            assert key == DOCKERFILE_LABEL
            return self.label

        def build_image(self, tag: str, dockerfile: object, ctx: object, labels: dict) -> None:  # type: ignore[type-arg]
            self.built.append(labels)

    stale, fresh, missing = Docker("0ld"), Docker(current), Docker(None)
    for d in (stale, fresh, missing):
        ensure_image(d, repo)  # type: ignore[arg-type]
    assert stale.built == [{DOCKERFILE_LABEL: current}]
    assert fresh.built == []
    assert missing.built == [{DOCKERFILE_LABEL: current}]


def test_a_saved_error_is_labelled_when_it_is_handed_back(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """After a repos.yaml fix, validate printed the old error as if it had just happened."""
    import asyncio

    from eval_harness import paths
    from eval_harness.cli import _reused_note
    from eval_harness.harness import runner
    from eval_harness.harness.record import AttemptRecord, load_record, save_record

    monkeypatch.setenv("ABEVAL_DATA_ROOT", str(tmp_path))
    paths.reset_cache()
    rec = AttemptRecord(
        case_id="C-1",
        run_id="validate",
        model="noop",
        status="error",
        phase="setup",
        started_at="2026-09-30T21:59:00Z",
        error="npm: not found",
    )
    rec.finished_at = "2026-09-30T21:59:10Z"
    save_record(rec)
    monkeypatch.setattr(runner, "ensure_image", lambda docker, repo: None)
    monkeypatch.setattr(runner, "Docker", lambda: None)
    case = type("C", (), {"case_id": "C-1"})()
    [back] = asyncio.run(
        runner.run_many([case], None, adapter=None, caps=None, run_id="validate", concurrency=1)  # type: ignore[arg-type,list-item]
    )
    assert back.reused and "saved result from 2026-09-30T21:59:10Z" in _reused_note(back)
    assert "--retry-errors" in _reused_note(back)
    assert "reused" not in (load_record("validate", "C-1") or rec).model_dump_json()

    ran: list[str] = []

    async def fake_run_case(case, repo, **kw):  # type: ignore[no-untyped-def]
        ran.append(case.case_id)
        return rec

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    asyncio.run(
        runner.run_many(  # type: ignore[arg-type]
            [case], None, adapter=None, caps=None, run_id="validate", concurrency=1, recheck=True
        )
    )
    assert ran == ["C-1"], "--recheck must ignore the saved result"


def test_a_suite_that_collects_nothing_is_the_environment_not_the_case() -> None:
    """With the venv missing from PATH, validate marked a good case invalid, so run skipped it."""
    from eval_harness.harness.runner import broken_environment
    from eval_harness.harness.testrun import TestResult

    def result(total: int, code: int) -> TestResult:
        return TestResult(
            total=total,
            passed=total,
            failed=0,
            skipped=0,
            exit_code=code,
            output="/usr/local/bin/python: No module named pytest",
            failed_tests=[],
        )

    reason = broken_environment(result(0, 1))
    assert reason and "No module named pytest" in reason
    assert broken_environment(result(120, 1)) is None  # a suite with failures is a baseline
    assert broken_environment(result(0, 0)) is None
    assert broken_environment(None) is None
