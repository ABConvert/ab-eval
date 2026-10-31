"""Prompt-run metrics, compare, leaderboard, export and the dashboard task views."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from eval_harness import paths
from eval_harness.adapters.oneshot import OneshotResult
from eval_harness.harness.record import RunMeta, save_run_meta
from eval_harness.prompt.cases import load_prompt_cases
from eval_harness.prompt.metrics import (
    cohen_kappa,
    quality_of,
    self_consistency,
    weighted_kappa,
)
from eval_harness.prompt.report import compare, leaderboard, summary_json
from eval_harness.prompt.runner import run_prompt_cases
from eval_harness.prompt.score import PromptCaseScore, load_prompt_summary, score_prompt_run
from eval_harness.prompt.task import Dimension, Rubric, load_task
from tests.dashboard_client import TestClient
from tests.prompt_helpers import TASK, clear_spend_env, model, result, setup_root

SECRET_OUTPUT = "PRIVATE-OUTPUT-TEXT six reps at 4:05 interval run"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    clear_spend_env(monkeypatch)
    paths.reset_cache()
    setup_root(tmp_path)
    return tmp_path


def _judge_reply(faith: int, use: int, typ: str) -> str:
    return json.dumps(
        {
            "faithfulness": {"score": faith, "reason": f"REASON-TEXT {faith}"},
            "usefulness": {"score": use, "reason": "fine"},
            "type_correct": {"score": typ, "reason": "ok"},
        }
    )


async def _make_run(
    run_id: str,
    *,
    model_key: str = "cand",
    faith: dict[str, int] | None = None,
    repeats: int = 1,
    dataset: str | None = "all",
    cost: float = 0.01,
) -> None:
    """Generate and score one run with fakes: judge-a says `faith`, judge-b one lower."""
    cases = load_prompt_cases(TASK)
    task = load_task(TASK)
    save_run_meta(
        RunMeta(
            run_id=run_id,
            model=model_key,
            dataset=dataset,
            kind="prompt",
            task=TASK,
            repeats=repeats,
            model_config_snapshot=model(model_key).model_dump(),
            caps={},
            concurrency=1,
            harness_git_sha="abc",
            image="",
            cases=[c.case_id for c in cases],
            started_at=f"2026-10-31T{len(run_id):02d}:00",
        )
    )

    async def gen(prompt: str, **kw: Any) -> OneshotResult:
        case = next(c for c in cases if c.prompt == prompt)
        return result(f"{SECRET_OUTPUT} {case.case_id}", cost=cost)

    await run_prompt_cases(
        cases, model(model_key), task=task, run_id=run_id, repeats=repeats, generate=gen
    )
    faith = faith or {}

    async def judges(prompt: str, *, model: Any, **kw: Any) -> OneshotResult:
        cid = next(c.case_id for c in cases if c.case_id in prompt)
        f = faith.get(cid, 4)
        if model.key == "judge-b":
            f = max(1, f - 1)
        return result(_judge_reply(f, 3, "yes"), cost=0.001)

    await score_prompt_run(
        run_id, task=task, judges=[model("judge-a"), model("judge-b")], generate=judges
    )


# ------------------------------------------------------------------ statistics


def test_cohen_kappa_matches_a_hand_computed_value() -> None:
    # po = 0.75; pe = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = (0.75-0.5)/(1-0.5)
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == 0.5
    assert cohen_kappa(["y", "n"], ["y", "n"]) == 1.0
    assert cohen_kappa(["y", "y"], ["y", "y"]) is None, "all one label: kappa is undefined"
    assert cohen_kappa(["y"], ["y"]) is None


def test_weighted_kappa_counts_a_near_miss_as_partly_right() -> None:
    a = [1, 2, 3, 4, 5]
    assert weighted_kappa(a, a, 1, 5) == 1.0
    near = weighted_kappa(a, [2, 3, 4, 5, 5], 1, 5)
    far = weighted_kappa(a, [5, 4, 3, 2, 1], 1, 5)
    assert near is not None and far is not None
    assert near > 0.5 > far, "off-by-one is close agreement; reversed is disagreement"
    assert far < 0


def _row(**kw: Any) -> PromptCaseScore:
    base: dict[str, Any] = {"case_id": "c", "repeat": 0, "split": "dev", "status": "completed"}
    base.update(kw)
    return PromptCaseScore.model_validate(base)


def test_quality_is_a_weighted_normalised_composite() -> None:
    rubric = Rubric(
        dimensions=[
            Dimension(name="a", scale="1-5", instruction="i", weight=2),
            Dimension(name="b", scale=["yes", "no"], good=["yes"], instruction="i"),
            Dimension(name="c", scale=["x", "y"], instruction="i"),  # no good label: unranked
        ]
    )
    row = _row(judge_mean={"a": 5.0, "b": 0.0}, scorer_metrics={"ok": True, "len": 900.0})
    # (2 * 1.0 + 1 * 0.0 + 1 * 1.0) / 4: `ok` is in quality_metrics, `len` is not.
    assert quality_of(row, rubric, {"ok": 1.0}) == 0.75
    assert quality_of(_row(status="score_error"), rubric, {}) is None, "unknown, not zero"
    assert quality_of(_row(status="error"), rubric, {}) is None


def test_self_consistency_measures_movement_between_repeats() -> None:
    rubric = Rubric(dimensions=[Dimension(name="a", scale="1-5", instruction="i")])
    rows = [
        _row(case_id="c1", repeat=0, judge_mean={"a": 4.0}),
        _row(case_id="c1", repeat=1, judge_mean={"a": 2.0}),
        _row(case_id="c2", repeat=0, judge_mean={"a": 3.0}),
        _row(case_id="c2", repeat=1, judge_mean={"a": 3.0}),
    ]
    sc = self_consistency(rows, rubric, {})
    assert sc["cases"] == 2 and sc["a"]["mean_abs_diff"] == 1.0
    assert self_consistency(rows[::2], rubric, {}) == {}, "one repeat each: nothing to compare"


# ------------------------------------------------------------------ summary metrics


async def test_summary_metrics_cover_dimensions_strata_agreement_and_cost(root: Path) -> None:
    await _make_run("r1", faith={"demo-sum-0001": 5, "demo-sum-0002": 2, "demo-sum-0003": 4})
    m = load_prompt_summary("r1").metrics
    faith = m["dimensions"]["faithfulness"]
    assert faith["n"] == 3 and faith["mean"] == pytest.approx((4.5 + 1.5 + 3.5) / 3, abs=1e-3)
    assert faith["ci"] is not None and faith["judge_disagreement_mean"] == 1.0
    assert m["dimensions"]["type_correct"]["distribution"] == {"yes": 1.0}
    assert m["scorer"]["word_count"]["kind"] == "number"
    assert m["scorer"]["names_type"]["kind"] == "bool"
    assert set(m["strata"]) >= {"paired_plan", "detected_ne_final", "long", "split:dev"}
    assert m["strata"]["split:holdout"]["n"] == 1
    pair = m["agreement"]["judge-a|judge-b"]["faithfulness"]
    assert pair["kind"] == "quadratic_weighted" and pair["n"] == 3
    assert m["cost"]["per_case_usd"] == pytest.approx(0.01)
    assert m["latency"]["p50_s"] == 0.5 and m["tokens"]["output"] == 300
    assert m["quality"]["n"] == 3 and m["error_rate"] == 0.0


async def test_repeats_produce_self_consistency(root: Path) -> None:
    await _make_run("rep", repeats=2)
    m = load_prompt_summary("rep").metrics
    assert m["self_consistency"]["cases"] == 3
    assert m["self_consistency"]["faithfulness"]["mean_abs_diff"] == 0.0


# ------------------------------------------------------------------ compare and leaderboard


async def test_compare_pairs_attempts_and_reports_per_dimension_differences(root: Path) -> None:
    await _make_run("base", faith={"demo-sum-0001": 2, "demo-sum-0002": 2, "demo-sum-0003": 2})
    await _make_run("better", faith={"demo-sum-0001": 5, "demo-sum-0002": 5, "demo-sum-0003": 5})
    cmp = compare([load_prompt_summary("base"), load_prompt_summary("better")])
    o = cmp["against"][0]
    assert o["paired"] == 3
    assert o["dimensions"]["faithfulness"]["diff"] == 3.0
    assert o["dimensions"]["usefulness"]["diff"] == 0.0
    assert o["quality"]["diff"] > 0 and "word_count" in o["scorer"]


async def test_compare_refuses_runs_of_different_tasks(root: Path) -> None:
    await _make_run("a1")
    s = load_prompt_summary("a1")
    other = s.model_copy(update={"task": "other", "run_id": "x"})
    with pytest.raises(ValueError, match="different tasks"):
        compare([s, other])


async def test_leaderboard_ranks_by_quality_within_one_dataset(root: Path) -> None:
    await _make_run("low", faith={"demo-sum-0001": 1, "demo-sum-0002": 1, "demo-sum-0003": 1})
    await _make_run("high", faith={"demo-sum-0001": 5, "demo-sum-0002": 5, "demo-sum-0003": 5})
    await _make_run("adhoc", dataset=None)
    boards = leaderboard([load_prompt_summary(r) for r in ("low", "high", "adhoc")])
    assert [b["dataset"] for b in boards] == ["(ad hoc)", "all"], "never ranked across datasets"
    rows = boards[1]["rows"]
    assert [r["run_id"] for r in rows] == ["high", "low"] and rows[0]["rank"] == 1
    assert rows[0]["cost_per_case_usd"] == pytest.approx(0.01)
    assert rows[0]["quality_per_dollar"] is not None and rows[0]["latency_p90_s"] == 0.5


# ------------------------------------------------------------------ export


async def test_summary_json_carries_numbers_and_no_case_text(root: Path) -> None:
    await _make_run("r1")
    body = json.dumps(summary_json(load_prompt_summary("r1")))
    for text in ("PRIVATE-OUTPUT-TEXT", "REASON-TEXT", "Activity data", "running coach"):
        assert text not in body, f"{text!r} leaked into the export"
    data = json.loads(body)
    assert data["format"] == "ab-eval/prompt-summary@1" and data["task"] == TASK
    assert data["metrics"]["quality"]["n"] == 3
    assert data["cases"][0]["judge_mean"]["faithfulness"] == 3.5
    assert "instruction" not in json.dumps(data["rubric"])


def test_cli_export_compare_and_leaderboard(root: Path) -> None:
    from eval_harness.cli import app

    asyncio.run(_make_run("c1", faith={"demo-sum-0001": 2}))
    asyncio.run(_make_run("c2"))
    runner = CliRunner()
    out = runner.invoke(app, ["export", "--run", "c1", "--format", "summary-json"])
    assert out.exit_code == 0, out.output
    exported = root / "results" / "c1" / "export.summary.json"
    assert "PRIVATE-OUTPUT-TEXT" not in exported.read_text()
    out = runner.invoke(app, ["compare", "--runs", "c1", "--runs", "c2"])
    assert out.exit_code == 0, out.output
    md = (root / "results" / "compare-prompt-c1-vs-c2.md").read_text()
    assert "rubric: faithfulness" in md
    out = runner.invoke(app, ["leaderboard", "--task", TASK])
    assert out.exit_code == 0, out.output
    assert "c2" in out.output.splitlines()[3], "the better run ranks first"
    bad = runner.invoke(app, ["export", "--run", "c1", "--format", "csv"])
    assert bad.exit_code != 0


# ------------------------------------------------------------------ dashboard


def test_dashboard_task_views_render(root: Path) -> None:
    from eval_harness.dashboard.app import build_app

    asyncio.run(_make_run("d1"))
    asyncio.run(_make_run("d2", faith={"demo-sum-0001": 1}))
    client = TestClient(build_app())
    page = client.get("/tasks")
    assert page.status_code == 200 and TASK in page.text
    page = client.get(f"/tasks/{TASK}")
    assert page.status_code == 200 and "Leaderboard" in page.text and "d1" in page.text
    page = client.get(f"/tasks/{TASK}/runs/d1")
    assert page.status_code == 200
    assert "PRIVATE-OUTPUT-TEXT" in page.text, "the run view shows the output"
    assert "REASON-TEXT" in page.text, "and the judges' reasons beside it"
    assert "synthetic" in page.text, "and the grading context"
    one = client.get(f"/tasks/{TASK}/runs/d1?case=demo-sum-0002")
    assert "demo-sum-0001" not in one.text.split("Attempts", 1)[1]
    page = client.get(f"/tasks/{TASK}/compare?runs=d1&runs=d2")
    assert page.status_code == 200 and "paired attempts" in page.text
    assert client.get("/tasks/no-such-task").status_code == 404
    assert client.get(f"/tasks/{TASK}/runs/nope").status_code == 404


def test_dashboard_escapes_model_output(root: Path) -> None:
    from eval_harness.dashboard.app import build_app
    from eval_harness.prompt.record import load_attempt, save_attempt

    asyncio.run(_make_run("x1"))
    rec = load_attempt("x1", "demo-sum-0001", 0)
    assert rec is not None
    rec.output_text = "<script>alert(1)</script>"
    save_attempt(rec)
    page = TestClient(build_app()).get(f"/tasks/{TASK}/runs/x1")
    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;" in page.text
