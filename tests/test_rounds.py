"""Synthetic round lifecycle and frozen execution regressions; no model or Docker calls."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from eval_harness import paths, rounds
from eval_harness.cli import app
from eval_harness.collect.cases import Case
from eval_harness.dashboard import jobs, round_pages
from eval_harness.dashboard.app import build_app
from eval_harness.harness.record import RunMeta, save_run_meta
from eval_harness.report.summary import CaseScore, RunSummary
from tests.conftest import FIXTURES
from tests.dashboard_client import TestClient
from tests.test_dashboard import _case


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(FIXTURES / "config", tmp_path / "config")
    case_dir = tmp_path / "data" / "cases"
    case_dir.mkdir(parents=True)
    (case_dir / "DEMO-1.json").write_text(json.dumps({**_case("DEMO-1"), "repo": "acme/demo-app"}))
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    return tmp_path


def test_frozen_cases_configs_append_retry_and_duplicate(workspace: Path) -> None:
    round_ = rounds.create("October", "all")
    original = round_.model_dump(mode="json")
    entry, rid = rounds.append_models(round_.id, ["demo-api"])[0]
    (workspace / "data/cases/DEMO-1.json").unlink()
    (workspace / "config/repos.yaml").unlink()
    second, second_rid = rounds.append_models(round_.id, ["demo-compat"])[0]
    assert rid != second_rid and second.model.key == "demo-compat"
    frozen, candidate = rounds.execution(round_.id, rid)
    assert frozen.cases[0].task_prompt == original["cases"][0]["task_prompt"]
    assert frozen.repo.model_dump(mode="json") == original["repo"]
    assert candidate.model == entry.model
    retry_entry, retry_id = rounds.retry(round_.id, entry.id)
    assert retry_id != rid and retry_entry.run_ids == [rid, retry_id]
    with pytest.raises(ValueError, match="already belongs"):
        rounds.append_models(round_.id, ["demo-api"])
    duplicate = rounds.create("October copy", "", source=round_.id)
    assert duplicate.id != round_.id and duplicate.cases == frozen.cases
    assert not duplicate.entries
    assert rounds.load(round_.id).entries[0].run_ids == [rid, retry_id]


def test_concurrent_additions_do_not_lose_models(workspace: Path) -> None:
    round_ = rounds.create("Concurrent", "all")
    with ThreadPoolExecutor(2) as executor:
        list(
            executor.map(
                lambda key: rounds.append_models(round_.id, [key]), ["demo-api", "demo-compat"]
            )
        )
    assert len(rounds.load(round_.id).entries) == 2


def test_invalid_selection_does_not_change_round(workspace: Path) -> None:
    round_ = rounds.create("Safe", "all")
    for keys in (["judge"], ["missing"], ["demo-api", "demo-api"]):
        with pytest.raises(ValueError):
            rounds.append_models(round_.id, keys)
    assert not rounds.load(round_.id).entries
    with pytest.raises(ValueError):
        rounds.load("../../config")


def test_frozen_cli_does_not_read_changed_cases_or_configs(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval_harness.adapters import registry
    from eval_harness.harness import runner

    round_ = rounds.create("Execution", "all")
    entry, rid = rounds.append_models(round_.id, ["demo-api"])[0]
    shutil.rmtree(workspace / "config")
    shutil.rmtree(workspace / "data")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(registry, "make_adapter", lambda key, models: seen.update(models=models))

    async def run_many(cases: list[Case], repo: Any, **kwargs: Any) -> list[Any]:
        seen.update(cases=cases, repo=repo, **kwargs)
        return []

    monkeypatch.setattr(runner, "run_many", run_many)
    result = CliRunner().invoke(
        app, ["run", "--model", "ignored", "--round-id", round_.id, "--run-id", rid, "--no-score"]
    )
    assert result.exit_code == 0, result.output
    assert seen["cases"] == round_.cases and seen["repo"] == round_.repo
    assert seen["models"] == {entry.model.key: entry.model}
    assert seen["caps"] == round_.caps
    assert not seen["retry_errors"]
    meta = json.loads((workspace / "results" / rid / "run.json").read_text())
    assert meta["round_id"] == round_.id
    repeated = CliRunner().invoke(
        app, ["run", "--model", "ignored", "--round-id", round_.id, "--run-id", rid, "--no-score"]
    )
    assert repeated.exit_code != 0


def _scored(round_: rounds.Round, entry: rounds.Entry, rid: str) -> None:
    case = CaseScore(
        case_id="DEMO-1",
        kind="bug_fix",
        tier="small",
        author_kind="human",
        spec_level="specific",
        split="holdout",
        status="completed",
        resolved=True,
        tests_pass=1,
        no_regression=1,
        lint_clean=1,
        patch_similarity=1,
        code_quality=0.8,
        cost_usd=1,
        input_tokens=100,
        output_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        wall_clock_seconds=100,
        agent_wall_clock_seconds=90,
        turns=2,
        tool_calls=2,
        cap_hit=None,
        touched_test_files=False,
        host_exec_items=0,
        files_touched=1,
        diff_lines=2,
    )
    meta = RunMeta(
        run_id=rid,
        round_id=round_.id,
        model=entry.model.key,
        model_config_snapshot=entry.model.model_dump(),
        caps=round_.caps.model_dump(),
        concurrency=1,
        harness_git_sha=round_.harness_sha,
        image=round_.repo.image,
        cases=["DEMO-1"],
        started_at="2026-10-03T00:00:00Z",
        finished_at="2026-10-03T00:01:00Z",
    )
    save_run_meta(meta)
    summary = RunSummary(
        run_id=rid,
        model=entry.model.key,
        model_config_snapshot=entry.model.model_dump(),
        caps=round_.caps.model_dump(),
        harness_git_sha=round_.harness_sha,
        judge=round_.judge.model,
        judge_config=round_.judge.model_dump(),
        scored_at="now",
        cases=[case],
    )
    (paths.results_root() / rid / "summary.json").write_text(summary.model_dump_json())


def test_round_rankings_use_compatible_attempts_and_keep_previous_while_pending(
    workspace: Path,
) -> None:
    first = rounds.create("First", "all")
    second = rounds.create("Second", "all")
    entry, rid = rounds.append_models(first.id, ["demo-api"])[0]
    other, other_rid = rounds.append_models(second.id, ["demo-compat"])[0]
    _scored(first, entry, rid)
    _scored(second, other, other_rid)
    assert [r.run_id for r in round_pages.eligible(rounds.load(first.id))] == [rid]
    rounds.retry(first.id, entry.id)
    assert [r.run_id for r in round_pages.eligible(rounds.load(first.id))] == [rid]
    summary_path = paths.results_root() / rid / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["caps"]["max_turns"] = 500
    summary_path.write_text(json.dumps(summary))
    assert not round_pages.eligible(rounds.load(first.id))


def test_round_pages_and_queue_are_scoped(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs.JobQueue, "_loop", lambda self: None)
    queue = jobs.JobQueue(workspace / "results/_jobs")
    monkeypatch.setattr(jobs, "_queue", queue)
    client = TestClient(build_app())
    response = client.post(
        "/rounds/create", data={"name": "October", "dataset": "all", "models": "demo-api"}
    )
    assert response.status_code in (200, 303), response.text
    round_ = rounds.all_rounds()[0]
    entry = round_.entries[0]
    assert len(queue.queued()) == 1
    assert "--round-id" in queue.queued()[0].argv
    added = client.post(f"/rounds/{round_.id}/models", data={"models": "demo-compat"})
    assert added.status_code in (200, 303)
    assert len(queue.queued()) == 2
    assert rounds.load(round_.id).entries[0].run_ids == entry.run_ids
    for path in (
        "/rounds",
        "/rounds/new",
        f"/rounds/{round_.id}",
        f"/rounds/{round_.id}/compare",
        f"/rounds/{round_.id}/leaderboard",
        "/setup?step=repository",
        "/setup?step=model",
        "/setup?step=ready",
    ):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
    response = client.post(f"/rounds/{round_.id}/retry/{entry.id}")
    assert response.status_code == 409
    queue.shutdown()


def test_round_scoring_uses_frozen_judge_and_cases(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval_harness.cli import _score
    from eval_harness.dashboard import data
    from eval_harness.metrics import dataset

    round_ = rounds.create("Judge snapshot", "all")
    entry, rid = rounds.append_models(round_.id, ["demo-api"])[0]
    _scored(round_, entry, rid)
    summary = data.load_run(rid).summary
    shutil.rmtree(workspace / "config")
    seen = {}

    def score(run_id: str, **kwargs: Any) -> RunSummary:
        seen.update(kwargs)
        return summary

    monkeypatch.setattr(dataset, "score_run", score)
    _score(rid, judge=True, cache=True)
    assert seen["judge_cfg"] == round_.judge
    assert seen["cases_dir"] == rounds.directory(round_.id) / "cases"
    with pytest.raises(Exception, match="frozen judge"):
        _score(rid, judge=False, cache=True)


def test_changed_evaluator_refuses_append_and_human_reference_is_not_ranked(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_ = rounds.create("Controls", "all")
    control, control_rid = rounds.append_models(round_.id, ["human-patch"])[0]
    _scored(round_, control, control_rid)
    assert not round_pages.eligible(rounds.load(round_.id))
    monkeypatch.setattr(rounds, "harness_git_sha", lambda: "different")
    with pytest.raises(ValueError, match="evaluator changed"):
        rounds.append_models(round_.id, ["demo-api"])
    with pytest.raises(ValueError, match="version differs"):
        rounds.execution(round_.id, control_rid)
