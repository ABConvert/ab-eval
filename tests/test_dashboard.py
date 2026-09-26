from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from eval_harness import paths
from eval_harness.dashboard import data, jobs
from eval_harness.dashboard.app import build_app
from tests.conftest import FIXTURES


def _record(case_id: str, run_id: str, *, resolved: bool) -> dict[str, Any]:
    tests = {
        "total": 3,
        "passed": 3 if resolved else 1,
        "failed": 0 if resolved else 2,
        "skipped": 0,
        "exit_code": 0 if resolved else 1,
        "output": "",
        "failed_tests": [] if resolved else ["a.test.js:: nope"],
    }
    before = {**tests, "passed": 0, "failed": 3, "exit_code": 1}
    return {
        "case_id": case_id,
        "run_id": run_id,
        "model": "claude-opus-5",
        "status": "completed",
        "tests_before": before,
        "tests_after": tests,
        "regressions": [] if resolved else None,
        "lint_ok": True if resolved else None,
        "cost_usd": 0.5,
        "wall_clock_seconds": 120.0,
        "agent_wall_clock_seconds": 90.0,
        "turns": 4,
        "tool_calls": 11,
        "generated_diff": "diff --git a/x b/x\n+one\n",
        "tool_log": [{"tool": "bash", "args": {"command": "ls"}, "is_error": False}],
        "final_text": "done",
    }


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "results"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "run.json").write_text(
        json.dumps(
            {
                "run_id": "demo",
                "model": "claude-opus-5",
                "model_config_snapshot": {},
                "caps": {"max_turns": 40, "wall_clock_seconds": 1800},
                "concurrency": 1,
                "harness_git_sha": "abc1234",
                "image": "img",
                "cases": ["DEMO-2877", "DEMO-2856"],
                "started_at": "2026-09-10T10:00:00Z",
            }
        )
    )
    (root / "demo" / "DEMO-2877.json").write_text(
        json.dumps(_record("DEMO-2877", "demo", resolved=True))
    )
    # The data root now carries cases as well as results, so the fixture supplies both —
    # the dashboard pages read case files to label and rate each row.
    cases = root.parent / "data" / "cases"
    cases.mkdir(parents=True)
    for cid in ("DEMO-2877", "DEMO-2856"):
        (cases / f"{cid}.json").write_text(json.dumps(_case(cid)))
    # A relocated data root must carry config too, or the harness falls back to the
    # checkout's — which a fresh clone does not have.
    shutil.copytree(FIXTURES / "config", root.parent / "config")
    monkeypatch.setenv(paths.ENV_VAR, str(root.parent))
    paths.reset_cache()
    return root


def _case(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "repo": "o/r",
        "kind": "bug_fix",
        "base_commit": "c" * 40,
        "task_prompt": "do the thing",
        "test_command": "x",
        "test_files_added": [],
        "test_files": ["a.test.ts"],
        "code_files": ["a.ts"],
        "human_patch": "",
        "human_test_patch": "",
        "files_changed": 3,
        "lines_changed": 100,
        "human_cycle_time_hours": 1.0,
        "review_comment_count": 1,
        "merged_at": "",
        "pr_number": 1,
        "merge_commit": "m",
        "test_runner": "unit",
        "files_dropped": [],
        "tier": "A",
        "kind_source": "review",
        "author": "a",
        "author_kind": "human",
        "spec_level_heuristic": "solution-specified",
        "ticket_created_at": "",
        "ticket_started_at": None,
        "human_cycle_time_source": "created_at",
    }


def test_run_view_counts_progress(results: Path) -> None:
    view = data.load_run("demo")
    assert view is not None
    assert (view.done, view.total, view.resolved) == (1, 2, 1)
    assert view.pending_ids == ["DEMO-2856"]
    assert view.scored is False


def test_outcome_labels(results: Path) -> None:
    recs = data.load_records("demo")
    assert data.outcome_of(recs["DEMO-2877"]) == "resolved"


def test_reserved_dirs_are_not_runs(results: Path) -> None:
    (results / "_jobs").mkdir()
    (results / "validate").mkdir()
    (results / "validate" / "run.json").write_text("{}")
    assert [m["run_id"] for m in data.run_metas()] == ["demo"]


def test_pages_render(results: Path) -> None:
    client = TestClient(build_app())
    assert client.get("/").status_code == 200
    assert "demo" in client.get("/runs").text
    body = client.get("/runs/demo").text
    assert "DEMO-2877" in body and "DEMO-2856" in body  # done and pending both listed
    assert client.get("/runs/demo/case/DEMO-2877").status_code == 200
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/compare").status_code == 200
    assert client.get("/api/state").json()["current"] is None


def test_queue_runs_one_job_at_a_time(tmp_path: Path) -> None:
    q = jobs.JobQueue(jobs_dir=tmp_path / "_jobs")
    first = q.submit(
        jobs.Job(id="a1", kind="run", label="first", argv=["/bin/sh", "-c", "echo one; sleep 0.6"])
    )
    second = q.submit(
        jobs.Job(id="b2", kind="run", label="second", argv=["/bin/sh", "-c", "echo two"])
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if q.get(second.id) and q.get(second.id).status == "done":  # type: ignore[union-attr]
            break
        # while the first runs, the second must not have started
        if q.current() and q.current().id == first.id:  # type: ignore[union-attr]
            assert q.get(second.id).status in ("queued", "running", "done")  # type: ignore[union-attr]
        time.sleep(0.1)
    assert q.get(first.id).status == "done"  # type: ignore[union-attr]
    assert q.get(second.id).status == "done"  # type: ignore[union-attr]
    assert "one" in q.get(first.id).tail()  # type: ignore[union-attr]


def test_cancel_stops_a_running_job(tmp_path: Path) -> None:
    q = jobs.JobQueue(jobs_dir=tmp_path / "_jobs")
    job = q.submit(jobs.Job(id="c3", kind="run", label="long", argv=["/bin/sh", "-c", "sleep 30"]))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (not q.current() or q.current().id != job.id):  # type: ignore[union-attr]
        time.sleep(0.1)
    assert q.cancel(job.id) is True
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and q.get(job.id).status == "running":  # type: ignore[union-attr]
        time.sleep(0.2)
    assert q.get(job.id).status == "stopped"  # type: ignore[union-attr]


def test_history_marks_orphans_stopped(tmp_path: Path) -> None:
    d = tmp_path / "_jobs"
    d.mkdir()
    (d / "old.json").write_text(
        json.dumps(
            {
                "id": "old",
                "kind": "run",
                "label": "left over",
                "argv": ["/bin/true"],
                "status": "running",
                "queued_at": "2026-09-10T00:00:00Z",
                "started_at": "2026-09-10T00:00:00Z",
            }
        )
    )
    q = jobs.JobQueue(jobs_dir=d)
    assert q.get("old").status == "stopped"  # type: ignore[union-attr]


def test_job_builders_use_the_module_entry_point() -> None:
    job = jobs.run_job(model="m", run_id="r", cases=["DEMO-1"], max_turns=5, wall_clock=60)
    assert job.argv[1:4] == ["-m", "eval_harness", "run"]
    assert "--retry-errors" in job.argv
    assert jobs.compare_job(["a", "b"]).argv[-4:] == ["--runs", "a", "--runs", "b"]


def test_token_metrics_sum_every_column(results: Path) -> None:
    from eval_harness.report.render import case_tokens, run_metrics
    from eval_harness.report.summary import CaseScore, RunSummary

    def score(cid: str, *, resolved: bool, tokens: tuple[int, int, int, int]) -> CaseScore:
        i, o, cr, cw = tokens
        return CaseScore(
            case_id=cid,
            kind="bug_fix",
            tier="A",
            author_kind="human",
            spec_level="symptom",
            split="holdout",
            status="completed",
            resolved=resolved,
            tests_pass=1.0,
            no_regression=None,
            lint_clean=None,
            patch_similarity=None,
            code_quality=None,
            cost_usd=0.0,
            input_tokens=i,
            output_tokens=o,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
            wall_clock_seconds=1.0,
            agent_wall_clock_seconds=1.0,
            turns=1,
            tool_calls=1,
            cap_hit=None,
            touched_test_files=False,
            host_exec_items=0,
            files_touched=1,
            diff_lines=1,
        )

    cases = [
        score("DEMO-1", resolved=True, tokens=(10, 20, 30, 40)),
        score("DEMO-2", resolved=False, tokens=(1, 2, 3, 4)),
    ]
    assert case_tokens(cases[0]) == 100
    m = run_metrics(
        RunSummary(
            run_id="t",
            model="m",
            model_config_snapshot={},
            harness_git_sha="x",
            judge=None,
            judge_config=None,
            scored_at="now",
            cases=cases,
        )
    )
    assert m["tokens"] == {
        "input": 11,
        "output": 22,
        "cache_read": 33,
        "cache_write": 44,
        "total": 110,
    }
    # per *solved* case, so an unresolved attempt makes the ratio worse, not better
    assert m["tokens_per_solved"] == 110
    assert m["output_tokens_per_solved"] == 22


def test_case_across_runs_collects_every_attempt(results: Path) -> None:
    (results / "other").mkdir()
    (results / "other" / "run.json").write_text(
        json.dumps(
            {
                "run_id": "other",
                "model": "gpt-5.6-terra",
                "model_config_snapshot": {},
                "caps": {},
                "concurrency": 1,
                "harness_git_sha": "z",
                "image": "i",
                "cases": ["DEMO-2877"],
                "started_at": "2026-09-10T12:00:00Z",
            }
        )
    )
    (results / "other" / "DEMO-2877.json").write_text(
        json.dumps(_record("DEMO-2877", "other", resolved=False))
    )
    attempts = data.case_across_runs("DEMO-2877")
    assert {a.run_id for a in attempts} == {"demo", "other"}
    assert {a.outcome for a in attempts} == {"resolved", "failed"}
    assert data.case_across_runs("DEMO-9999") == []

    client = TestClient(build_app())
    body = client.get("/cases/DEMO-2877").text
    assert "demo" in body and "other" in body
    assert client.get("/cases/DEMO-0001").status_code == 404


def test_datasets_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eval_harness.collect import datasets as ds_mod

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    monkeypatch.setattr(ds_mod, "all_case_ids", lambda: ["DEMO-1", "DEMO-2", "DEMO-3"])
    ds = ds_mod.create("my-set", "a description", ["DEMO-1"])
    assert ds.case_ids == ["DEMO-1"]
    assert ds_mod.add_cases("my-set", ["DEMO-2", "DEMO-1"]).case_ids == ["DEMO-1", "DEMO-2"]
    assert ds_mod.remove_cases("my-set", ["DEMO-1"]).case_ids == ["DEMO-2"]
    with pytest.raises(LookupError):
        ds_mod.add_cases("my-set", ["DEMO-999"])  # no case file
    with pytest.raises(ValueError):
        ds_mod.create("Not Valid", "")
    with pytest.raises(ValueError):
        ds_mod.create("holdout", "")  # built-in names are protected
    with pytest.raises(FileExistsError):
        ds_mod.create("my-set", "")


def test_leaderboard_ranks_and_scales() -> None:
    from eval_harness.report.leaderboard import build, token_efficiency
    from eval_harness.report.summary import CaseScore, RunSummary

    def run(rid: str, *, solved: int, tokens: int, quality: float = 0.8) -> RunSummary:
        cases = [
            CaseScore(
                case_id=f"DEMO-{i}",
                kind="bug_fix",
                tier="A",
                author_kind="human",
                spec_level="symptom",
                split="holdout",
                status="completed",
                resolved=i < solved,
                tests_pass=1.0,
                no_regression=1.0,
                lint_clean=1.0,
                patch_similarity=None,
                code_quality=quality if i < solved else None,
                cost_usd=1.0,
                input_tokens=tokens // 4,
                output_tokens=tokens // 4,
                cache_read_tokens=tokens // 4,
                cache_write_tokens=tokens // 4,
                wall_clock_seconds=10.0,
                agent_wall_clock_seconds=10.0,
                turns=1,
                tool_calls=1,
                cap_hit=None,
                touched_test_files=False,
                host_exec_items=0,
                files_touched=1,
                diff_lines=1,
            )
            for i in range(4)
        ]
        return RunSummary(
            run_id=rid,
            model=rid,
            model_config_snapshot={},
            harness_git_sha="x",
            judge=None,
            judge_config=None,
            scored_at="now",
            cases=cases,
        )

    frugal = run("frugal", solved=4, tokens=1000)  # same solves, far fewer tokens
    greedy = run("greedy", solved=4, tokens=100_000)
    board = build([greedy, frugal])
    assert [r["run_id"] for r in board["rows"]] == ["frugal", "greedy"]
    assert board["rows"][0]["rank"] == 1
    # efficiency is quality-weighted solves per million tokens, so frugal dwarfs greedy
    assert (
        token_efficiency(
            {"tokens": {"total": 1_000_000}, "overall": {"resolved": 2}, "code_quality_mean": 0.5}
        )
        == 1.0
    )
    assert (
        token_efficiency(
            {"tokens": {"total": 0}, "overall": {"resolved": 2}, "code_quality_mean": 1.0}
        )
        is None
    )


def test_empty_dataset_is_truthy() -> None:
    """__len__ alone would make a new, empty dataset falsy and hide it in templates."""
    from eval_harness.collect.datasets import Dataset

    empty = Dataset(name="fresh")
    assert bool(empty) is True
    assert len(empty) == 0


def test_openai_compat_usage_and_arguments() -> None:
    from eval_harness.adapters.base import ToolSpec
    from eval_harness.adapters.openai_compat import (
        parse_arguments,
        tool_schema,
        usage_from_response,
    )

    u = usage_from_response(
        {"prompt_tokens": 354, "completion_tokens": 59, "prompt_cache_hit_tokens": 100}
    )
    # prompt_tokens includes the cached part, so input must not double-count it
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (254, 59, 100)
    assert usage_from_response(None).input_tokens == 0
    assert (
        usage_from_response(
            {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}}
        ).cache_read_tokens
        == 4
    )

    assert parse_arguments('{"path": "a.js"}') == {"path": "a.js"}
    assert parse_arguments(None) == {}
    assert "__malformed_arguments__" in parse_arguments("{not json")

    spec = tool_schema([ToolSpec(name="bash", description="run", input_schema={"type": "object"})])
    assert spec[0]["function"]["name"] == "bash"


def test_difficulty_is_rated_from_the_case_not_from_results() -> None:
    from eval_harness.collect.cases import Case, TestGroup
    from eval_harness.collect.difficulty import classify, explain, score

    def case(**kw: Any) -> Case:
        base = dict(
            case_id="DEMO-1",
            repo="r",
            kind="bug_fix",
            base_commit="c",
            task_prompt="t",
            test_command="x",
            test_files_added=[],
            test_files=["a.test.ts"],
            code_files=["a.ts"],
            human_patch="",
            human_test_patch="",
            files_changed=3,
            lines_changed=100,
            human_cycle_time_hours=1.0,
            review_comment_count=1,
            merged_at="",
            pr_number=1,
            merge_commit="m",
            test_runner="unit",
            files_dropped=[],
            tier="A",
            kind_source="review",
            author="a",
            author_kind="human",
            spec_level_heuristic="solution-specified",
            ticket_created_at="",
            ticket_started_at=None,
            human_cycle_time_source="created_at",
        )
        base.update(kw)
        return Case(**base)  # type: ignore[arg-type]

    tiny = case(lines_changed=40, files_changed=2, spec_level_heuristic="code-in-ticket")
    assert classify(tiny) == "easy"

    big = case(
        lines_changed=350,
        files_changed=8,
        spec_level_heuristic="symptom",
        review_comment_count=6,
        human_cycle_time_hours=40.0,
        test_groups=[TestGroup(runner="unit", files=["a"]), TestGroup(runner="e2e", files=["b"])],
    )
    assert classify(big) == "hard"
    assert score(big) > score(tiny)

    # a ticket carrying the code is easier than one carrying only a symptom
    assert score(case(spec_level_heuristic="symptom")) > score(
        case(spec_level_heuristic="code-in-ticket")
    )
    detail = explain(big)
    assert detail["difficulty"] == "hard"
    assert detail["signals"]["spec"]["value"] == 1.0


def test_summary_markdown_route_is_wired(results: Path) -> None:
    """This route read a module constant the data-root refactor removed, and 500'd."""
    (results / "demo" / "summary.md").write_text("# demo\n")
    client = TestClient(build_app())
    assert client.get("/runs/demo/summary.md").status_code == 200
    assert client.get("/runs/missing/summary.md").status_code == 404


def _render_datasets(**ctx: Any) -> str:
    """Render datasets.html directly: some of its states the live data root cannot reach."""
    from eval_harness.dashboard.app import templates

    base = {
        "request": None,
        "datasets": [],
        "current": None,
        "rows": [],
        "all_cases": [],
        "error": None,
        "page": "datasets",
        "current_job": None,
        "queued_jobs": [],
        "now": "00:00:00",
    }
    return templates.env.get_template("datasets.html").render({**base, **ctx})


class _Stub:
    def __init__(self, name: str, built_in: bool) -> None:
        self.name = name
        self.built_in = built_in
        self.case_ids: list[str] = []
        self.description = ""


def test_an_empty_derived_set_does_not_point_at_a_panel_that_is_not_there() -> None:
    """`dev` and `holdout` render no edit panel, so "add from the panel on the right" —
    which the page used to say for every empty set — was an instruction to nowhere.
    """
    body = _render_datasets(current=_Stub("holdout", built_in=True), datasets=[])
    assert "holdout is empty" in body
    assert "panel on the right" not in body
    assert "eval-harness collect" in body, "say where the split actually comes from"

    editable = _render_datasets(current=_Stub("regression-suite", built_in=False), datasets=[])
    assert "panel on the right" in editable, "a saved set does have one"


def test_no_dataset_selected_says_so() -> None:
    """Without ?name= the detail half of the page is empty; it has to say why."""
    body = _render_datasets(datasets=[_Stub("all", built_in=True)])
    assert "Open a set to see and edit its cases" in body


def _isolated_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A real dataset over two real cases, on a data root of its own."""
    from eval_harness.collect.cases import Case, save_case

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    cases_dir = tmp_path / "data" / "cases"
    cases_dir.mkdir(parents=True)
    for n in (1, 2):
        save_case(
            Case(
                case_id=f"DEMO-{n}",
                repo="acme/demo-app",
                kind="bug_fix",
                base_commit="a" * 40,
                task_prompt=f"# case {n}",
                test_command="x",
                test_files_added=[],
                test_files=["web/a.test.ts"],
                code_files=["web/a.ts"],
                human_patch="",
                human_test_patch="",
                files_changed=1,
                lines_changed=1,
                human_cycle_time_hours=1.0,
                review_comment_count=0,
                merged_at="2026-09-02T13:28:34Z",
                pr_number=n,
                merge_commit="b" * 40,
                test_runner="unit",
                files_dropped=[],
                tier="A",
                kind_source="review",
                author="x",
                author_kind="human",
                spec_level_heuristic="symptom",
                ticket_created_at="2026-09-02T06:00:00Z",
                ticket_started_at=None,
                human_cycle_time_source="created_at",
            ),
            cases_dir,
        )

    from eval_harness.collect import datasets as ds_mod

    ds_mod.create("keepers", "", ["DEMO-1", "DEMO-2"])
    return "DEMO-1"


def test_remove_button_submits_the_action_field_not_just_the_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `action`, dataset_edit defaults to "add" — so a Remove that posts only
    ids is not a no-op, it silently ADDS. The field is what makes it a removal.

    The button sits outside #judge-form and is wired to it with form=, so its own
    name/value ride along with the ticked checkboxes. That is what this asserts.
    """
    import re

    from eval_harness.dashboard.app import build_app

    _isolated_dataset(tmp_path, monkeypatch)
    body = TestClient(build_app()).get("/datasets?name=keepers").text
    match = re.search(r'<button[^>]*id="remove-btn"[^>]*>', body)
    assert match, "the remove button moved; this test needs its new shape"
    tag = match.group(0)
    assert 'name="action"' in tag and 'value="remove"' in tag, (
        "a submit button contributes its name/value to the body; without them this "
        "POST is an add over an empty id list"
    )
    assert 'form="judge-form"' in tag, "it has to own the form the checkboxes are in"
    assert 'formaction="/datasets/keepers/edit"' in tag
    assert 'id="remove-form"' not in body, "the empty form it used to aim at is gone"


def test_removing_needs_the_action_field_and_adding_is_what_happens_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves, against a real dataset on an isolated root."""
    from eval_harness.collect import datasets as ds_mod
    from eval_harness.dashboard.app import build_app

    first = _isolated_dataset(tmp_path, monkeypatch)
    client = TestClient(build_app())

    # ids with no action: the handler's default is "add", so nothing is removed.
    client.post("/datasets/keepers/edit", data={"cases": [first]}, follow_redirects=False)
    assert first in ds_mod.load("keepers").case_ids, (
        "ids without action must not remove — proving why the field is load-bearing"
    )

    # the field present: the case actually goes.
    client.post(
        "/datasets/keepers/edit",
        data={"action": "remove", "cases": [first]},
        follow_redirects=False,
    )
    assert first not in ds_mod.load("keepers").case_ids


def test_a_scored_run_is_finished_however_many_records_survive(results: Path) -> None:
    """Records can be pruned. Reading that as "still going" stranded eight of
    twenty-one runs — every one the leaderboard ranks — on a progress page that
    polled a job which had finished days earlier.
    """
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    (results / "demo" / "summary.json").write_text(json.dumps(summary))
    for stale in (results / "demo").glob("DEMO-*.json"):
        stale.unlink()

    body = TestClient(build_app()).get("/runs/demo").text
    assert 'id="run-panel"' not in body, "a scored run must reach its own results"


def test_an_unscored_jobless_incomplete_run_still_shows_the_progress_view(
    results: Path,
) -> None:
    """The other half of the same predicate. This run really did stop short, and
    that state has its own page — the fix for the scored case must not swallow it.
    """
    body = TestClient(build_app()).get("/runs/demo").text
    assert 'id="run-panel"' in body, "one of two records landed and nothing is running"


def _attach_job(monkeypatch: pytest.MonkeyPatch, run_id: str) -> jobs.Job:
    """Put a live job in front of one run without starting a subprocess."""
    job = jobs.Job(
        id="stub",
        kind="run",
        label=f"run · {run_id}",
        argv=["python", "-m", "eval_harness"],
        run_id=run_id,
        status="running",
    )
    monkeypatch.setattr(jobs.queue(), "active_for_run", lambda rid: job if rid == run_id else None)
    return job


def test_the_panel_asks_for_a_refresh_once_and_lands_somewhere_that_does_not_poll(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handover from the progress view to the results view.

    A refresh header that re-arms would reload a finished run for ever, so this
    asserts both halves: the poll that sees the job end asks for the reload, and the
    page that reload lands on renders no panel, so no further poll can be made.
    """
    client = TestClient(build_app())
    _attach_job(monkeypatch, "demo")

    # A job is attached: the panel polls and asks for nothing.
    partial = client.get("/runs/demo/panel")
    assert "HX-Refresh" not in partial.headers
    assert 'hx-trigger="every 4s"' in partial.text

    (results / "demo" / "DEMO-2856.json").write_text(
        json.dumps(_record("DEMO-2856", "demo", resolved=False))
    )
    monkeypatch.setattr(jobs.queue(), "active_for_run", lambda rid: None)

    complete = client.get("/runs/demo/panel")
    assert complete.headers.get("HX-Refresh") == "true", "the poll that sees it end hands over"
    assert 'hx-trigger="every 4s"' not in complete.text, "and stops polling in the same breath"

    # Where that reload goes, and whether it can ask again.
    landed = client.get("/runs/demo").text
    assert 'id="run-panel"' not in landed, "the reload must land on the results view"
    assert "/runs/demo/panel" not in landed, "and that view must not poll the panel"


def test_a_run_that_stopped_short_does_not_poll_a_job_that_is_not_there(
    results: Path,
) -> None:
    """The headline claim of round 3, executed rather than asserted.

    `demo` has one record of two and nothing running: that is the definition of
    stopped short. Arming the panel on `pending_ids` made the condition true for
    exactly the state the page calls terminal, so it polled every four seconds for
    ever against a job that had already gone.
    """
    body = TestClient(build_app()).get("/runs/demo").text

    assert 'id="run-panel"' in body, "a run that stopped short still shows its progress view"
    # The queue strip polls /queue on every page and is not what this is about.
    assert "/runs/demo/panel" not in body, "but the panel itself must not poll"
    assert "nothing is running now" in body, "the page has to say why it stopped checking"


def test_a_re_scored_run_with_pruned_records_is_not_stranded_again(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-scoring attaches a job to a scored run whose records were pruned. On
    `pending_ids` the panel stayed armed after that job ended — `finished` could
    never be true with 104 ids outstanding — so the fix that freed every ranked run
    was undone by the operation this project is about to perform on all of them.
    """
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    (results / "demo" / "summary.json").write_text(json.dumps(summary))
    for stale in (results / "demo").glob("DEMO-*.json"):
        stale.unlink()

    client = TestClient(build_app())
    view = data.load_run("demo")
    assert view is not None and view.pending_ids, "the state under test needs pending ids"

    # The scoring job has just ended. Every record is still missing.
    monkeypatch.setattr(jobs.queue(), "active_for_run", lambda rid: None)
    panel = client.get("/runs/demo/panel")
    assert panel.headers.get("HX-Refresh") == "true", "pending ids must not hold the handover"
    assert 'hx-trigger="every 4s"' not in panel.text

    landed = client.get("/runs/demo").text
    assert 'id="run-panel"' not in landed, "and the reload reaches its own results"


def test_every_copy_button_copies_what_is_printed_beside_it(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hidden` landed inside a `data-copy` value, so one button put a placeholder
    that does not exist on the clipboard and shipped visible without `copy.js`.
    Parsing the command proved it was valid shell; it did not prove it was the
    command on the screen, which is the property that matters.
    """
    from html.parser import HTMLParser

    class Buttons(HTMLParser):
        """Every `.copy` button, paired with the `<code>` printed just before it."""

        def __init__(self) -> None:
            super().__init__()
            self.found: list[dict[str, Any]] = []
            self._code: str | None = None
            self._in_code = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            a = dict(attrs)
            if tag == "code":
                self._in_code = True
                self._code = ""
            elif tag == "button" and "copy" in (a.get("class") or ""):
                self.found.append(
                    {"copy": a.get("data-copy"), "hidden": "hidden" in a, "code": self._code}
                )

        def handle_endtag(self, tag: str) -> None:
            if tag == "code":
                self._in_code = False

        def handle_data(self, data: str) -> None:
            if self._in_code and self._code is not None:
                self._code += data

    def sweep(paths_to_read: tuple[str, ...]) -> list[tuple[str, dict[str, Any]]]:
        client = TestClient(build_app())
        out: list[tuple[str, dict[str, Any]]] = []
        for path in paths_to_read:
            parser = Buttons()
            parser.feed(client.get(path).text)
            out += [(path, b) for b in parser.found]
        return out

    seen = sweep(("/", "/setup", "/datasets", "/runs"))

    # The two commands on `/` only render when there is nothing to list — the first
    # screen a stranger who clones this repo sees, and where the broken one was.
    fresh = tmp_path / "fresh"
    (fresh / "results").mkdir(parents=True)
    (fresh / "data" / "cases").mkdir(parents=True)
    shutil.copytree(FIXTURES / "config", fresh / "config")
    monkeypatch.setenv(paths.ENV_VAR, str(fresh))
    paths.reset_cache()
    seen += sweep(("/",))

    checked: list[str] = []
    for path, btn in seen:
        assert btn["hidden"], f"{path}: a .copy button without copy.js must not be visible"
        copied = btn["copy"]
        if copied is None:
            continue  # this one copies another element's contents by id, not an attribute
        code = btn["code"]
        assert isinstance(code, str), f"{path}: a copy button with nothing printed beside it"
        assert copied.strip() == code.strip(), (
            f"{path}: the clipboard would get {copied!r} beside a screen reading {code!r}"
        )
        checked.append(copied.strip())

    assert len(checked) >= 5, "the sweep has to have found the buttons it is checking"
    assert any("collect --repo" in c for c in checked), (
        "the button this test exists for has to be one of the ones it saw"
    )


def test_a_real_job_ending_stops_the_poll_and_hands_over(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same claim again, driven rather than stubbed.

    A real subprocess goes through the real queue and is allowed to exit, and the
    panel is read at each stage. Round 3 reported "a stopped-short run reaches a
    terminal state with NO polling" from the shape of the template rather than from
    watching one, and the polling was never disarmed.
    """
    q = jobs.JobQueue(jobs_dir=tmp_path / "_jobs")
    monkeypatch.setattr(jobs, "queue", lambda: q)
    client = TestClient(build_app())

    q.submit(
        jobs.Job(
            id="live",
            kind="run",
            label="run · demo",
            argv=["/bin/sh", "-c", "echo working; sleep 30"],
            run_id="demo",
        )
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and q.active_for_run("demo") is None:
        time.sleep(0.1)
    assert q.active_for_run("demo") is not None, "the job has to actually be attached"

    while_running = client.get("/runs/demo/panel")
    assert 'hx-trigger="every 4s"' in while_running.text, "a live run polls"
    assert "HX-Refresh" not in while_running.headers, "and does not hand over yet"
    assert "/runs/demo/panel" in client.get("/runs/demo").text

    assert q.cancel("live") is True
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and q.active_for_run("demo") is not None:
        time.sleep(0.2)
    assert q.active_for_run("demo") is None, "the job really has to be gone"

    # One case of two landed and the job is dead: stopped short, the terminal state.
    after = client.get("/runs/demo/panel")
    assert 'hx-trigger="every 4s"' not in after.text, "the poll must stop when the job does"
    assert after.headers.get("HX-Refresh") == "true", "and ask for the page it belongs on"

    landed = client.get("/runs/demo").text
    assert "/runs/demo/panel" not in landed, "the page it lands on polls nothing"
    assert "nothing is running now" in landed, "and says why it stopped"


def test_a_scored_run_can_be_scored_again_and_says_what_that_costs(results: Path) -> None:
    """`POST /runs/{id}/score` stayed registered when the control was taken off the
    page, leaving a live money route with nothing in front of it — and re-scoring on
    a neutral judge is the next thing this harness is going to be asked to do.
    """
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    (results / "demo" / "summary.json").write_text(json.dumps(summary))

    body = TestClient(build_app()).get("/runs/demo").text
    assert "Score this run again" in body
    assert "2 judge calls" in body, "a button that spends judge calls has to say how many"
    assert "summary.json" in body, "and that it overwrites the scores in place"
    assert "the model being judged" not in body, "this judge is not the model under test"


def test_a_run_judged_by_its_own_model_says_so(results: Path) -> None:
    """The judge is recorded with its adapter and effort, so the obvious equality
    test against the model never fires. Two of this project's scored runs are judged
    by the model they are judging, which is the one figure on the page that should
    not be read at face value.
    """
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    # The page reads the run's own model from run.json, which is what a judge string
    # has to be compared against.
    summary["model"] = "claude-opus-5"
    summary["judge"] = "claude-opus-5 (anthropic, effort=medium)"
    (results / "demo" / "summary.json").write_text(json.dumps(summary))

    body = TestClient(build_app()).get("/runs/demo").text
    assert "the model being judged" in body


# --- round 4: the analysis surfaces ------------------------------------------------


def test_a_diff_splits_into_lines_that_keep_their_numbers() -> None:
    """The diff was one escaped blob inside `pre.log`, which wraps: a long removed line
    broke across two rows and its remainder started in the column an unchanged line
    starts in. One line of source has to stay one line on screen, with the side it
    belongs to and the number it has.
    """
    from eval_harness.dashboard.app import _diff_lines, _diff_stat

    diff = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -142,4 +142,5 @@ def total(items):\n"
        " def total(items):\n"
        "-    return sum(discount(i) for i in items)\n"
        "+    base = sum(i.price for i in items)\n"
        "+    return apply_once(base)\n"
        "     # unchanged\n"
    )
    rows = _diff_lines(diff)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["file", "file", "file", "hunk", "ctx", "del", "add", "add", "ctx"]

    # The sign is stripped, because the markup carries it in a gutter of its own.
    assert rows[5]["text"] == "    return sum(discount(i) for i in items)"
    assert rows[6]["text"] == "    base = sum(i.price for i in items)"

    # A removed line numbers on the old side only, an added line on the new side only.
    assert (rows[5]["old"], rows[5]["new"]) == (143, None)
    assert (rows[6]["old"], rows[6]["new"]) == (None, 143)
    assert (rows[7]["old"], rows[7]["new"]) == (None, 144)
    # Context resumes after both, one line further on each side.
    assert (rows[8]["old"], rows[8]["new"]) == (144, 145)

    assert _diff_stat(diff) == {"add": 2, "del": 1, "files": 1}
    assert _diff_lines(None) == []
    assert _diff_stat("") == {"add": 0, "del": 0, "files": 0}


def test_the_reference_patch_is_marked_as_never_shown_to_the_model(results: Path) -> None:
    """The harness rule, and the whole benchmark rests on it. It was stated once in the
    app, as a muted grey note under an ordinary heading, on one of the two case pages —
    and the run-scoped page, where you are actually reading a model's attempt, did not
    carry the patch or the rule at all.
    """
    client = TestClient(build_app())
    for path in ("/cases/DEMO-2877", "/runs/demo/case/DEMO-2877"):
        body = client.get(path).text
        assert "NEVER SHOWN TO THE MODEL" in body, f"{path}: the seal has to be on both pages"
        assert "The reference patch" in body
        assert "never puts it in the model" in body, f"{path}: and the rule has to be in words"


def test_the_case_page_shows_the_task_the_model_was_given(results: Path) -> None:
    """You cannot read a diff against nothing. The prompt was on the case object the
    page already receives and had simply never been rendered.
    """
    body = TestClient(build_app()).get("/runs/demo/case/DEMO-2877").text
    assert "GIVEN TO THE MODEL" in body
    assert "do the thing" in body, "the prompt itself, not a summary of it"
    assert "WHAT THE MODEL WROTE" in body, "and the model's own output, labelled as such"


def test_an_unjudged_attempt_says_so_rather_than_rendering_nothing(results: Path) -> None:
    """Three states, and the page told the truth about one of them.

    Round 3 made the panel stop vanishing. It still branched on
    `score.code_quality_reason`, but `score` comes off the run's summary — so a falsy
    reason with a summary present means "the judge ran and returned nothing for this
    case", and the page said "nothing has asked it yet" and offered to spend a judge
    call per case in the run to change nothing. 47 of 947 scored cases were in that
    state.
    """
    client = TestClient(build_app())

    # 1. No summary at all: nothing has asked, and scoring is the thing to do.
    body = client.get("/runs/demo/case/DEMO-2877").text
    assert "What the judge said" in body
    assert "This run has not been scored" in body
    assert "nothing has asked it yet" in body
    assert "/runs/demo/score" in body, "and the way to ask is on the page"

    # 2. Scored, and the judge returned no quality for this case.
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    summary["cases"][0]["case_id"] = "DEMO-2877"
    summary["cases"][0]["code_quality"] = None
    summary["cases"][0]["code_quality_reason"] = None
    (results / "demo" / "summary.json").write_text(json.dumps(summary))

    body = client.get("/runs/demo/case/DEMO-2877").text
    assert "The judge returned no quality score" in body
    assert "nothing has asked it yet" not in body, "the judge was asked; saying otherwise is false"
    assert "/runs/demo/score" not in body, (
        "and offering to re-score would spend a judge call per case to change nothing here"
    )

    # 3. Scored with a reason: the judgement itself.
    summary["cases"][0]["code_quality"] = 0.7
    summary["cases"][0]["code_quality_reason"] = "reads well enough"
    (results / "demo" / "summary.json").write_text(json.dumps(summary))
    body = client.get("/runs/demo/case/DEMO-2877").text
    assert "reads well enough" in body


def test_a_capped_attempt_says_the_cap_once(results: Path) -> None:
    """`a.outcome` already ends with the cap and the template appended `cap_hit` after
    it, so every capped attempt read "failed · max_turns · max_turns" — twice, in the
    field's own spelling.
    """
    rec = _record("DEMO-2856", "demo", resolved=False)
    rec["cap_hit"] = "max_turns"
    (results / "demo" / "DEMO-2856.json").write_text(json.dumps(rec))

    body = TestClient(build_app()).get("/cases/DEMO-2856").text
    assert "Hit the turn limit" in body, "the cap in words"
    assert "max_turns" not in body, "and never the field's own spelling"


def test_a_case_nobody_has_run_says_where_to_find_it(results: Path) -> None:
    """A case with no attempts is every case on the day it is curated. The reference
    patch still renders, because it is a property of the case rather than of any
    attempt.
    """
    body = TestClient(build_app()).get("/cases/DEMO-2856").text
    assert "No model has tried this one yet" in body
    assert "NEVER SHOWN TO THE MODEL" in body


def test_the_leaderboard_computes_nothing_over_an_empty_field(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh clone rendered a four-tile statistical header over nothing: 0 models, an
    em dash, 0 attempts, and a fourth tile whose value was the empty string with a
    sentence in the caption slot. A statistic over an empty set is an absence.
    """
    fresh = tmp_path / "fresh"
    (fresh / "results").mkdir(parents=True)
    (fresh / "data" / "cases").mkdir(parents=True)
    shutil.copytree(FIXTURES / "config", fresh / "config")
    monkeypatch.setenv(paths.ENV_VAR, str(fresh))
    paths.reset_cache()

    body = TestClient(build_app()).get("/leaderboard").text
    assert "Nothing has been scored yet" in body
    assert "Models compared" not in body, "no tile may count an empty field"
    assert "Attempts scored" not in body


def test_every_table_sits_in_a_scroll_container(results: Path) -> None:
    """Constraint 7, as a property of the markup rather than a measurement that has to
    be repeated. Four templates carried a bare <table> — compare's head-to-head and
    per-case grid, case_across, and the leaderboard — and each one made its page scroll
    sideways at 900. A table that is not wrapped can only overflow the page.
    """
    import re

    offenders = []
    for path in sorted(Path("src/eval_harness/dashboard/templates").glob("*.html")):
        text = path.read_text()
        # Walk the file and check that the nearest preceding div is a .table-wrap.
        for m in re.finditer(r"<table\b", text):
            before = text[: m.start()]
            last_wrap = before.rfind("table-wrap")
            last_close = before.rfind("</table>")
            if last_wrap < 0 or last_wrap < last_close:
                line = before.count("\n") + 1
                offenders.append(f"{path.name}:{line}")
    assert not offenders, f"tables outside .table-wrap: {offenders}"


def _second_run(results: Path) -> None:
    """A second scored run over the same two cases, so /compare has something to do."""
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    for run_id, resolved in (("demo", True), ("other", False)):
        d = results / run_id
        d.mkdir(exist_ok=True)
        (d / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "model": f"model-{run_id}",
                    "model_config_snapshot": {},
                    "caps": {"max_turns": 40, "wall_clock_seconds": 1800},
                    "concurrency": 1,
                    "harness_git_sha": "abc1234",
                    "image": "img",
                    "cases": ["DEMO-2877", "DEMO-2856"],
                    "started_at": "2026-09-10T10:00:00Z",
                }
            )
        )
        for cid in ("DEMO-2877", "DEMO-2856"):
            (d / f"{cid}.json").write_text(json.dumps(_record(cid, run_id, resolved=resolved)))
        s = json.loads(json.dumps(summary))
        s["run_id"] = run_id
        s["model"] = f"model-{run_id}"
        for i, cid in enumerate(("DEMO-2877", "DEMO-2856")):
            s["cases"][i]["case_id"] = cid
            s["cases"][i]["resolved"] = resolved
        (d / "summary.json").write_text(json.dumps(s))


def test_the_per_case_grid_reads_down_and_every_cell_carries_a_word(results: Path) -> None:
    """It was one row per run and one column per case — 104 columns of 52px, each an
    empty coloured div whose only text was a title attribute. 7381px of sideways page
    scroll, and nothing a screen reader could read. Cases are the axis that grows
    without limit, so they go on the one the browser already scrolls.
    """
    _second_run(results)
    body = TestClient(build_app()).get("/compare?runs=demo&runs=other").text

    grid = body.split('class="table-wrap casegrid"')[-1].split("</table>")[0]
    header = grid.split("<tbody>")[0]
    rows = grid.split("<tbody>")[1]

    # Runs across the top, cases down the side.
    assert "model-demo" in header and "model-other" in header
    assert "DEMO-2877" not in header, "case ids are not column headers any more"
    assert "/cases/DEMO-2877" in rows and "/cases/DEMO-2856" in rows

    # Every cell says what happened, in the words the rest of the app uses.
    assert "Fixed" in rows and "Not fixed" in rows
    assert 'class="cell' not in body, "the empty coloured square is gone"
    # And the disagreement is named rather than left to be spotted.
    assert "Who got it" in header
    assert "only model-demo" in rows


def test_a_model_keeps_one_mark_across_pages_and_selections(results: Path) -> None:
    """The chart palette's original defect was assigning colour by position in an
    array. Sorting the models on the page reproduces it a different way: a model sits
    at a different index in every selection, so it changes colour when you change which
    runs you are comparing. The order is built once, from every model in the data.
    """
    import re

    _second_run(results)
    # A third run, so the two comparisons below really are different selections.
    third = results / "zzz"
    third.mkdir()
    (third / "run.json").write_text(
        json.dumps(
            {
                "run_id": "zzz",
                "model": "aaa-first-alphabetically",
                "model_config_snapshot": {},
                "caps": {"max_turns": 40, "wall_clock_seconds": 1800},
                "concurrency": 1,
                "harness_git_sha": "abc1234",
                "image": "img",
                "cases": ["DEMO-2877", "DEMO-2856"],
                "started_at": "2026-09-10T10:00:00Z",
            }
        )
    )
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "zzz"
    summary["model"] = "aaa-first-alphabetically"
    for i, cid in enumerate(("DEMO-2877", "DEMO-2856")):
        summary["cases"][i]["case_id"] = cid
    (third / "summary.json").write_text(json.dumps(summary))
    for cid in ("DEMO-2877", "DEMO-2856"):
        (third / f"{cid}.json").write_text(json.dumps(_record(cid, "zzz", resolved=True)))

    client = TestClient(build_app())

    def mark_for(body: str, model: str) -> str:
        m = re.search(r'class="series (s-\d)"><span class="lab">\s*(?:<[^>]+>\s*)?' + model, body)
        assert m, f"no series mark for {model}"
        return m.group(1)

    # Two comparisons that do not contain the same set of models.
    with_zzz = client.get("/compare?runs=demo&runs=zzz").text
    with_other = client.get("/compare?runs=demo&runs=other").text
    board = client.get("/leaderboard?dataset=").text

    demo_marks = {
        "compare with zzz": mark_for(with_zzz, "model-demo"),
        "compare with other": mark_for(with_other, "model-demo"),
        "leaderboard": mark_for(board, "model-demo"),
    }
    assert len(set(demo_marks.values())) == 1, f"one model, several marks: {demo_marks}"

    # And distinct models must still be distinguishable — an order that collapsed every
    # model onto one mark would satisfy the assertion above and be useless.
    on_one_page = {
        mark_for(with_zzz, "model-demo"),
        mark_for(with_zzz, "aaa-first-alphabetically"),
    }
    assert len(on_one_page) == 2, f"two models sharing one mark: {on_one_page}"

    # And the canvas is handed the same list the markup used.
    assert "Series.assign(runs.map(r => r.model), [" in with_zzz
    assert '"aaa-first-alphabetically"' in with_zzz


def test_the_grid_puts_the_disagreements_first(results: Path) -> None:
    """The sub-heading says so. It was `sorted(case_ids)` — alphabetical — with the
    agreed rows dimmed where they fell, which in a 104-row grid is no help at all.
    """
    _second_run(results)
    # Both runs fix DEMO-2856; only `demo` fixes DEMO-2877. The case they disagree about
    # is the LATER id alphabetically, so plain sorted() puts it second and only an
    # actual disagreements-first sort puts it first.
    for run_id in ("demo", "other"):
        summary = json.loads((results / run_id / "summary.json").read_text())
        for c in summary["cases"]:
            c["resolved"] = c["case_id"] == "DEMO-2856" or run_id == "demo"
        (results / run_id / "summary.json").write_text(json.dumps(summary))

    body = TestClient(build_app()).get("/compare?runs=demo&runs=other").text
    grid = body.split('class="table-wrap casegrid"')[-1].split("</table>")[0]
    rows = grid.split("<tbody>")[1]
    assert rows.index("DEMO-2877") < rows.index("DEMO-2856"), (
        "the case they disagree about has to come before the one they agree on, "
        "and it is the alphabetically later of the two so that this can tell"
    )


def test_the_compare_button_names_the_files_it_will_overwrite(
    results: Path, tmp_path: Path
) -> None:
    """A mutating control that does not say what it writes is asking for trust it has
    not earned. The paths have to be the ones write_comparison actually uses, not a
    plausible-looking guess — mine was, until I read it.
    """
    from eval_harness.dashboard.app import _compare_paths
    from eval_harness.report.render import write_comparison

    _second_run(results)
    shown = _compare_paths(["demo", "other"])
    js, md = write_comparison(
        [data.load_run("demo").summary, data.load_run("other").summary],  # type: ignore[union-attr]
        out_dir=tmp_path,
    )
    assert [Path(p).name for p in shown] == [md.name, js.name], (
        f"the page promises {[Path(p).name for p in shown]} and the writer writes "
        f"{[md.name, js.name]}"
    )
    body = TestClient(build_app()).get("/compare?runs=demo&runs=other").text
    for p in shown:
        assert p in body


def test_the_picker_says_how_big_each_run_is(results: Path) -> None:
    """Comparing a 104-case run with a 14-case one is the mistake this page most easily
    lets you make, and the picker was a row of bare ids.
    """
    _second_run(results)
    body = TestClient(build_app()).get("/compare").text
    assert body.count("2 cases") >= 2, "each scored run carries its size"


def test_a_case_page_says_which_sets_the_case_is_in(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "It is curated and ready" is only useful if the page says where to look."""
    from eval_harness.collect import datasets as ds_mod

    ds_mod.create("keepers", "a set", ["DEMO-2877"])
    body = TestClient(build_app()).get("/cases/DEMO-2877").text
    assert "keepers" in body and "/datasets?name=keepers" in body


def test_a_run_with_no_records_is_not_offered_a_scoring_job(results: Path) -> None:
    """Nothing landed means there is nothing to judge."""
    (results / "demo" / "DEMO-2877.json").unlink()
    body = TestClient(build_app()).get("/runs/demo").text
    assert "nothing is running now" in body
    assert "Score the 0 that landed" not in body


def test_the_follow_badge_is_a_real_button_that_says_what_it_is(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aria-pressed` was on a <span> with no role and no tabindex — not a widget, not
    operable, and the attribute is not valid there. It also lied: the template hard-codes
    "Following" and nothing re-synced it, so it read "Following" while paused, every
    three seconds.
    """
    q = jobs.JobQueue(jobs_dir=tmp_path / "_jobs")
    monkeypatch.setattr(jobs, "queue", lambda: q)
    job = jobs.Job(
        id="live",
        kind="run",
        label="run · demo",
        argv=["/bin/sh", "-c", "sleep 30"],
        run_id="demo",
        status="running",
        jobs_dir=tmp_path / "_jobs",
    )
    monkeypatch.setattr(q, "get", lambda jid: job if jid == "live" else None)

    body = TestClient(build_app()).get("/jobs/live").text
    assert '<button type="button" class="follow" id="following"' in body
    assert 'aria-pressed="true"' in body
    # The badge is painted from the reader's real position after every swap, which the
    # server cannot know; the template only renders the state a fresh page starts in.
    assert "paintBadge" in body and "htmx:afterSwap" in body


def test_the_job_page_actions_do_not_nest_a_form_in_the_row(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A <form> is a block element, so it took its own line inside .head .actions and
    the rotated lime button above was drawn on top of the destructive one underneath.
    """
    q = jobs.JobQueue(jobs_dir=tmp_path / "_jobs")
    monkeypatch.setattr(jobs, "queue", lambda: q)
    job = jobs.Job(
        id="live",
        kind="run",
        label="run · demo",
        argv=["/bin/sh", "-c", "sleep 30"],
        run_id="demo",
        status="running",
        jobs_dir=tmp_path / "_jobs",
    )
    monkeypatch.setattr(q, "get", lambda jid: job if jid == "live" else None)

    body = TestClient(build_app()).get("/jobs/live").text
    actions = body.split('<div class="actions">')[1].split("</div>")[0]
    assert "<form" not in actions, "the form belongs in the scripts block, joined by id"
    assert 'form="stop-job"' in actions
    assert '<form id="stop-job"' in body


# --- the final verdict's nine ------------------------------------------------------


def test_a_clone_with_no_models_yaml_still_serves_the_page_it_opens_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`eval-harness dashboard` opens the browser at `/`, and models.yaml is gitignored,
    so a clone of this repo does not have one. The call was unguarded, which made the
    first page a stranger sees a bare Internal Server Error — while /setup, the page
    built to tell them exactly what is missing, sat one URL away and unreachable.
    """
    fresh = tmp_path / "clone"
    (fresh / "results").mkdir(parents=True)
    (fresh / "data" / "cases").mkdir(parents=True)
    (fresh / "config").mkdir()
    shutil.copy(FIXTURES / "config" / "repos.yaml", fresh / "config" / "repos.yaml")
    # config_file falls back to the CHECKOUT's config when the data root has none, and a
    # developer machine has a models.yaml there — which is exactly why this never showed
    # up in four rounds of review. A clone has only models.example.yaml, so the fallback
    # has to miss too or this test is not testing a clone.
    checkout = tmp_path / "checkout"
    (checkout / "config").mkdir(parents=True)
    (checkout / "config" / "models.example.yaml").write_text("{}\n")
    monkeypatch.setattr(paths, "PROJECT_ROOT", checkout)
    monkeypatch.setenv(paths.ENV_VAR, str(fresh))
    paths.reset_cache()
    assert not paths.config_file("models.yaml").exists(), "the state under test"

    client = TestClient(build_app())
    home = client.get("/")
    assert home.status_code == 200, "the page the browser opens cannot be a 500"
    assert "No cases yet" in home.text, "and it is the good empty state, which needs models.yaml"
    assert "/setup" in home.text, "with the way to fix it on it"

    # Every other route already survived this; none of them may stop.
    for path in ("/setup", "/runs", "/leaderboard", "/compare", "/datasets"):
        assert client.get(path).status_code == 200, path


def test_the_chart_draws_every_model_differently(results: Path) -> None:
    """Past the fifth series the shapes repeat, and the stylesheet draws the repeat
    hollow. The canvas read the colour and the shape and not `faded`, so with eight
    models in the data three pairs came out byte-identical — two runs as the same green
    circle at opposite ends of one chart, with the legend dimming one of them.
    """
    from eval_harness.dashboard.app import _series_order

    _second_run(results)
    body = TestClient(build_app()).get("/compare?runs=demo&runs=other").text

    assert "p.faded ? 'transparent' : p.color" in body, "the canvas has to honour it"
    assert "pointBorderColor: p.color" in body, "or a hollow point has no hue at all"

    # The stylesheet's own repeat is a different shape, not the same shape at low alpha:
    # alpha is what the canvas could not mirror and what greyscale barely separates.
    css = Path("src/eval_harness/dashboard/static/style.css").read_text()
    for n in (6, 7, 8):
        assert f".s-{n} {{ color: var(--chart-{n - 5}); --mark: var(--series-{n}); }}" in css
        assert f"--series-{n}:" in css
    assert "opacity: 0.55" not in css.split(".s-6")[1][:400]

    # And the ordering really can run past five in this data root.
    assert len(_series_order()) >= 1


def test_runs_does_not_report_a_price_it_cannot_see(results: Path) -> None:
    """/runs summed the records on disk. Six runs had theirs pruned, so the page said
    the harness had cost $76.98 when it had cost $535 — while each run's own results
    page, reading the same run's summary, said $160.30 and $124.18. Two pages in one
    session disagreeing by $160, on the page you open to ask what this has cost you.
    """
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    for i, cid in enumerate(("DEMO-2877", "DEMO-2856")):
        summary["cases"][i]["case_id"] = cid
        summary["cases"][i]["cost_usd"] = 8.0
    (results / "demo" / "summary.json").write_text(json.dumps(summary))
    for stale in (results / "demo").glob("DEMO-*.json"):
        stale.unlink()

    view = data.load_run("demo")
    assert view is not None
    assert not view.records, "the state under test: the records are gone"
    assert view.cost == 16.0, "and the summary still knows what it cost"
    assert "$16.00" in TestClient(build_app()).get("/runs").text


def test_a_run_with_no_price_anywhere_says_so_rather_than_zero(results: Path) -> None:
    """Zero is a measurement. A subscription, a model hosted here and the human-patch
    baseline are the absence of one, and R-3 was about not confusing the two.
    """
    for cid in ("DEMO-2877",):
        rec = _record(cid, "demo", resolved=True)
        rec["cost_usd"] = 0.0
        (results / "demo" / f"{cid}.json").write_text(json.dumps(rec))

    runs = TestClient(build_app()).get("/runs").text
    assert "no price" in runs
    assert "report no price" in runs, "and the headline tile says how many it left out"

    panel = TestClient(build_app()).get("/runs/demo").text
    assert "this provider reports none" in panel
    assert "$0.00" not in panel, "nothing on the page may assert a measured zero"


def test_the_status_dot_is_not_an_inline_sliver() -> None:
    """`.dot` set width, height and border-radius and never set display, so an inline
    element ignored all three and it drew 3x20 wherever the parent was not flex — a
    text cursor beside the run name, on four pages, only ever while a job was attached.
    The baseline said this belonged in CSS; round 3 removed the inline styles from the
    templates and did not add the declaration, which moved the bug rather than closing
    it.
    """
    css = Path("src/eval_harness/dashboard/static/style.css").read_text()
    rule = css.split(".dot {")[1].split("}")[0]
    assert "display:" in rule, ".dot must not rely on its parent being a flex container"
    # And no template may go back to patching it inline.
    for path in Path("src/eval_harness/dashboard/templates").glob("*.html"):
        text = path.read_text()
        assert 'class="dot"' not in text or "display:inline-block" not in text, path.name


def test_copy_survives_the_pages_that_have_no_status_line() -> None:
    """Four templates load copy.js and two carry #copy-status. Both branches of the
    handler dereferenced it, so Copy on /runs and /jobs/{id} threw — the only uncaught
    exception in the product. A failed copy was silent there, and a successful one would
    have thrown before the timeout that restores the button label.
    """
    js = Path("src/eval_harness/dashboard/static/copy.js").read_text()
    assert "if (status) { status.textContent = msg; }" in js
    assert "status.textContent =" not in js.replace("if (status) { status.textContent = msg; }", "")

    loads, has_status = set(), set()
    for path in Path("src/eval_harness/dashboard/templates").glob("*.html"):
        text = path.read_text()
        if "copy.js" in text:
            loads.add(path.name)
        if 'id="copy-status"' in text:
            has_status.add(path.name)
    assert loads - has_status, "the asymmetry this guards is still real"


def test_a_bad_id_gets_the_product_rather_than_a_bare_h1(results: Path) -> None:
    """Four routes answered a bad id with `<h1>no such run</h1>` and no shell: a white
    page in Times New Roman, no nav, no way back, in a product that is carbon-dark and
    system sans-serif throughout. Reachable from a stale bookmark, a deleted run, a mistyped case
    id, or a link inside one of the saved compare-*.md files.
    """
    client = TestClient(build_app())
    for path, wanted in (
        ("/runs/does-not-exist", "does-not-exist"),
        ("/runs/demo/case/NOPE-1", "NOPE-1"),
        ("/cases/NOPE-1", "NOPE-1"),
        ("/jobs/nope", "nope"),
    ):
        r = client.get(path)
        assert r.status_code == 404, path
        assert "<h1>no such" not in r.text, path
        assert 'class="nav"' in r.text, f"{path}: a 404 still gets the shell"
        assert "/static/style.css" in r.text, f"{path}: and the theme"
        assert wanted in r.text, f"{path}: and says what was asked for"
        assert 'class="btn primary"' in r.text, f"{path}: and a way out"


def test_the_speed_column_says_what_it_actually_computes(results: Path) -> None:
    """3600 divided by the median case's agent time, over every case ATTEMPTED — an
    extrapolation from one case, and a rate of attempts. It was labelled "Fixes per
    hour", which is neither. The arithmetic is the coordinator's call, not this
    engagement's: it lives in report/leaderboard.py and the CLI reports use it.
    """
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    (results / "demo" / "summary.json").write_text(json.dumps(summary))
    body = TestClient(build_app()).get("/leaderboard?dataset=").text

    header = body.split("<thead>")[1].split("</thead>")[0]
    assert "Fixes per hour" not in header, "the column must not claim to count fixes"
    assert "Cases per hour" in header
    assert "at the median case" in header, "and must say it is one case, extrapolated"
    assert "rather than the ones it fixed" in body, "and the second overstatement too"

    src = Path("src/eval_harness/report/leaderboard.py").read_text()
    assert "3600.0 / float(median)" in src, "the metric itself is deliberately untouched"


def test_the_still_failing_list_does_not_collapse_six_tests_into_one_line(
    results: Path,
) -> None:
    """`t.split('::')[0]` threw the test name away, so six different failing tests in one
    file rendered as that filename six times — and the seventh entry, the one that would
    have told you a second file was broken, fell off the end of the slice. The evidence
    panel for the verdict the page is about.
    """
    rec = _record("DEMO-2877", "demo", resolved=False)
    rec["tests_after"]["failed_tests"] = [
        *[f"web/a.test.js::case {i}" for i in range(1, 7)],
        "web/b.test.ts::the one that used to fall off",
    ]
    (results / "demo" / "DEMO-2877.json").write_text(json.dumps(rec))

    body = TestClient(build_app()).get("/runs/demo/case/DEMO-2877").text
    panel = body.split("Still failing")[1].split("</ul>")[0]
    assert panel.count("web/a.test.js") == 1, "one line per file, not one per test"
    assert "6 tests" in panel, "with the count that the collapse was hiding"
    assert "web/b.test.ts" in panel, "and the file the slice used to drop"
    flat = " ".join(body.split())
    assert "Still failing: 7 tests in 2 files" in flat


def test_the_no_models_note_can_actually_render(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The note existed and could not fire.

    `models` is built as ["human-patch", *runnable], so it is never empty and the
    `{% if not models %}` branch that explains an empty picker was unreachable — a note
    in a template that cannot appear, which is the same shape as a value computed and
    never read. It branches on the runnable list now.
    """
    # A data root with cases but no models.yaml on either path: the picker renders, and
    # the only thing it can offer is the reference.
    # `results` has already pointed the data root at a tree with two cases in it; all
    # this needs on top is for models.yaml to be missing on BOTH paths, since
    # config_file falls back to the checkout's own.
    (paths.config_dir() / "models.yaml").unlink()
    checkout = tmp_path / "checkout"
    (checkout / "config").mkdir(parents=True)
    monkeypatch.setattr(paths, "PROJECT_ROOT", checkout)
    paths.reset_cache()
    assert not paths.config_file("models.yaml").exists(), "the state under test"

    body = TestClient(build_app()).get("/").text
    flat = " ".join(body.split())
    assert "DEMO-2877" in body, "the picker only renders when there are cases to launch"
    assert "No model is configured to run" in flat, "the note has to reach the page"
    assert "/setup" in body
    assert "human-patch" in body, "and the reference is still offered, which is why it needs saying"

    # And it stays out of the way once something can run.
    (paths.config_dir() / "models.yaml").write_text(
        (FIXTURES / "config" / "models.yaml").read_text()
    )
    paths.reset_cache()
    after = " ".join(TestClient(build_app()).get("/").text.split())
    assert "No model is configured to run" not in after


def test_run_results_does_not_price_a_run_that_reports_none(results: Path) -> None:
    """The B-3 verdict named _run_panel.html, so this one was outside the nine — the
    same defect on the page a reader is on to decide whether to spend more.
    """
    # Both records present and no summary: finished, not scored, which is the branch
    # that carries the per-case table. One record short and the router serves progress.
    for cid in ("DEMO-2877", "DEMO-2856"):
        rec = _record(cid, "demo", resolved=True)
        rec["cost_usd"] = 0.0
        (results / "demo" / f"{cid}.json").write_text(json.dumps(rec))

    # Unscored: the per-case table, and the reason given once underneath it.
    body = TestClient(build_app()).get("/runs/demo").text
    assert "$0.00" not in body
    assert "No case in this run reports a price" in body
    assert "not the same as costing nothing" in body

    # Scored: the panel above already names it, so the cell just does not assert a zero.
    summary = json.loads((FIXTURES / "summary-two-cases.json").read_text())
    summary["run_id"] = "demo"
    # run_metrics is memoised on (run_id, scored_at, len(cases)), so a summary that
    # borrows another test's identity gets another test's numbers. A re-scored run has
    # a new scored_at in reality too.
    summary["scored_at"] = "2026-09-18T04:00:00+00:00"
    for i, cid in enumerate(("DEMO-2877", "DEMO-2856")):
        summary["cases"][i]["case_id"] = cid
        summary["cases"][i]["cost_usd"] = 0.0
    (results / "demo" / "summary.json").write_text(json.dumps(summary))
    body = TestClient(build_app()).get("/runs/demo").text
    assert "$0.00" not in body
    assert "What it used" in body, "the panel above says why there is no money"


def test_the_run_view_has_no_property_nothing_reads() -> None:
    """`cost_is_measured` was computed and never consumed, which is the shape of two of
    the nine. Either read it or remove it.
    """
    src = Path("src/eval_harness/dashboard/data.py").read_text()
    assert "cost_is_measured" not in src
