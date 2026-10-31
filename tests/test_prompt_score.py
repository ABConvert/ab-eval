"""Scoring a prompt run: the scorer protocol, the rubric judge, its cache, and the ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from eval_harness import paths
from eval_harness.adapters import oneshot as oneshot_mod
from eval_harness.adapters.oneshot import OneshotResult, SpendLimitReached
from eval_harness.config import ModelConfig
from eval_harness.prompt.cases import load_prompt_case, load_prompt_cases
from eval_harness.prompt.judge import build_judge_prompt, judge_attempt, parse_judgement
from eval_harness.prompt.record import PromptAttempt
from eval_harness.prompt.runner import run_prompt_cases
from eval_harness.prompt.score import load_prompt_summary, score_prompt_run, summary_path
from eval_harness.prompt.task import load_task
from tests.prompt_helpers import TASK, clear_spend_env, model, result, setup_root

GOOD = json.dumps(
    {
        "faithfulness": {"score": 4, "reason": "paces match"},
        "usefulness": {"score": 3, "reason": "one tip"},
        "type_correct": {"score": "yes", "reason": "says intervals"},
    }
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    clear_spend_env(monkeypatch)
    paths.reset_cache()
    setup_root(tmp_path)
    return tmp_path


class Judges:
    """Fake one-shot for judges: answers per judge model, counts calls, keeps prompts."""

    def __init__(self, replies: dict[str, Any] | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, prompt: str, *, model: ModelConfig, **kw: Any) -> OneshotResult:
        self.calls.append((model.key, prompt))
        reply = self.replies.get(model.key, GOOD)
        if callable(reply):
            reply = reply(prompt)
        if isinstance(reply, BaseException):
            raise reply
        return result(str(reply), cost=0.002)


async def _generate(texts: dict[str, str] | None = None, run_id: str = "r1") -> None:
    from eval_harness.harness.record import RunMeta, save_run_meta

    texts = texts or {}
    cases = load_prompt_cases(TASK)
    save_run_meta(
        RunMeta(
            run_id=run_id,
            model="cand",
            kind="prompt",
            task=TASK,
            model_config_snapshot=model("cand").model_dump(),
            caps={},
            concurrency=1,
            harness_git_sha="abc",
            image="",
            cases=[c.case_id for c in cases],
            started_at="2026-10-31",
        )
    )

    async def gen(prompt: str, **kw: Any) -> OneshotResult:
        case = next(c for c in cases if c.prompt == prompt)
        return result(texts.get(case.case_id, "An interval run: six reps at 4:05 per km, steady."))

    await run_prompt_cases(cases, model("cand"), task=load_task(TASK), run_id=run_id, generate=gen)


async def _score(judges: Judges, keys: tuple[str, ...] = ("judge-a", "judge-b"), **kw: Any) -> Any:
    return await score_prompt_run(
        "r1", task=load_task(TASK), judges=[model(k) for k in keys], generate=judges, **kw
    )


async def test_a_failed_scorer_marks_score_error_and_never_zero(root: Path) -> None:
    await _generate({"demo-sum-0002": "SCORER_CRASH"})
    summary = await _score(Judges())
    row = next(c for c in summary.cases if c.case_id == "demo-sum-0002")
    assert row.status == "score_error" and "exited 3" in (row.scorer_error or "")
    assert row.scorer_metrics == {}, "no metrics, rather than metrics of zero"
    others = [c for c in summary.cases if c.case_id != "demo-sum-0002"]
    assert all(c.status == "completed" and "word_count" in c.scorer_metrics for c in others)


# ------------------------------------------------------------------ the judge prompt


def test_the_judge_prompt_is_built_from_the_rubric(root: Path) -> None:
    task = load_task(TASK)
    rubric = task.load_rubric()
    case = load_prompt_case(TASK, "demo-sum-0001")
    text = build_judge_prompt(rubric, case, "THE OUTPUT")
    assert "<output>\nTHE OUTPUT\n</output>" in text
    assert '"tier": "synthetic"' in text, "faithfulness uses the grading context"
    assert "an integer from 1 to 5" in text and '"yes", "no"' in text
    assert case.system not in text, "no dimension asks for the prompt"


def test_grading_context_stays_out_when_no_dimension_uses_it(root: Path) -> None:
    rubric = load_task(TASK).load_rubric()
    for d in rubric.dimensions:
        d.uses = ["output"]
    text = build_judge_prompt(rubric, load_prompt_case(TASK, "demo-sum-0001"), "x")
    assert "synthetic" not in text and "<grading_context>" not in text


def test_parse_judgement_holds_every_value_to_its_scale(root: Path) -> None:
    rubric = load_task(TASK).load_rubric()
    scores = parse_judgement(rubric, "Here you go:\n```json\n" + GOOD + "\n```")
    assert scores["faithfulness"].value == 4 and scores["type_correct"].value == "yes"
    for bad in (
        GOOD.replace('"score": 4', '"score": 7'),
        GOOD.replace('"yes"', '"maybe"'),
        json.dumps({"faithfulness": {"score": 3}}),
    ):
        with pytest.raises(ValueError):
            parse_judgement(rubric, bad)


# ------------------------------------------------------------------ judging and its cache


async def test_two_judges_score_every_attempt_independently(root: Path) -> None:
    await _generate()
    judges = Judges({"judge-b": GOOD.replace('"score": 4', '"score": 2')})
    summary = await _score(judges)
    assert len(judges.calls) == 6
    row = summary.cases[0]
    assert row.judge_scores["judge-a"]["faithfulness"] == 4
    assert row.judge_scores["judge-b"]["faithfulness"] == 2
    assert row.judge_mean["faithfulness"] == 3.0 and row.judge_disagreement["faithfulness"] == 2
    assert row.judge_mean["type_correct"] == 1.0, "share of judges answering a good label"
    assert row.judge_reasons["judge-a"]["faithfulness"] == "paces match"
    assert all("judge-a" not in p and "paces match" not in p for _, p in judges.calls), (
        "a judge never sees another judge's scores"
    )
    assert summary_path("r1").exists() and load_prompt_summary("r1").rubric_hash


async def test_judgements_are_cached_by_rubric_and_output(root: Path) -> None:
    await _generate()
    await _score(Judges())
    again = Judges()
    await _score(again)
    assert again.calls == [], "same run, case, repeat, judge, rubric and output: cached"
    # Change the rubric: every judgement is re-asked.
    rubric_file = load_task(TASK).directory / "rubric.yaml"
    rubric_file.write_text(rubric_file.read_text().replace("do not reward", "never reward"))
    edited = Judges()
    await _score(edited)
    assert len(edited.calls) == 6
    nocache = Judges()
    await _score(nocache, use_cache=False)
    assert len(nocache.calls) == 6


async def test_a_judge_reply_that_will_not_parse_is_retried_once(root: Path) -> None:
    await _generate()
    replies = iter(["no json here", GOOD])
    judges = Judges({"judge-a": lambda p: next(replies)})
    task = load_task(TASK)
    att = PromptAttempt(
        case_id="demo-sum-0001",
        run_id="r1",
        model="cand",
        repeat=0,
        status="completed",
        output_text="x",
    )
    j = await judge_attempt(
        model("judge-a"),
        task.load_rubric(),
        load_prompt_case(TASK, "demo-sum-0001"),
        att,
        generate=judges,
    )
    assert j.error is None and j.scores["usefulness"].value == 3
    assert len(judges.calls) == 2 and "could not be used" in judges.calls[1][1]


async def test_a_judge_that_always_fails_is_an_error_and_is_asked_again(root: Path) -> None:
    await _generate()
    summary = await _score(Judges({"judge-a": RuntimeError("judge down")}))
    row = summary.cases[0]
    assert "judge down" in row.judge_errors["judge-a"]
    assert "judge-a" not in row.judge_scores and row.status == "completed", "judge-b answered"
    retry = Judges()
    await _score(retry)
    assert {k for k, _ in retry.calls} == {"judge-a"}, "an error is never served from cache"


async def test_every_judge_failing_is_a_score_error(root: Path) -> None:
    await _generate()
    summary = await _score(Judges({"judge-a": "garbage", "judge-b": "garbage"}))
    assert {c.status for c in summary.cases} == {"score_error"}
    assert summary.cases[0].judge_mean == {}


async def test_a_judge_that_is_the_candidate_is_flagged(root: Path) -> None:
    await _generate()
    summary = await _score(Judges(), keys=("judge-a", "judge-self"))
    assert summary.self_preference == ["judge-self"]


async def test_an_errored_attempt_is_not_judged(root: Path) -> None:
    await _generate()
    from eval_harness.prompt.record import load_attempt, save_attempt

    rec = load_attempt("r1", "demo-sum-0003", 0)
    assert rec is not None
    rec.status, rec.output_text, rec.error = "error", "", "boom"
    save_attempt(rec)
    judges = Judges()
    summary = await _score(judges)
    assert len(judges.calls) == 4
    row = next(c for c in summary.cases if c.case_id == "demo-sum-0003")
    assert row.status == "error" and row.scorer_metrics == {}


# ------------------------------------------------------------------ ledger and limit, for real


class _Client:
    """httpx.AsyncClient stand-in for an openai-compat judge, so the real oneshot runs."""

    calls: ClassVar[int] = 0

    def __init__(self, **kw: Any) -> None:
        pass

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, *, headers: Any, json: Any) -> httpx.Response:
        _Client.calls += 1
        body = {
            "choices": [{"message": {"content": GOOD}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        }
        return httpx.Response(200, json=body)


def _priced_judge() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "key": "judge-a",
            "provider": "openai-compat",
            "model": "judge-a-1",
            "base_url": "https://example.invalid/v1",
            "price_per_mtok": {"input": 0.5, "output": 1.0, "cache_read": 0.0, "cache_write": 0.0},
        }
    )


async def test_judge_spend_goes_to_the_ledger(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _generate()
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    ledger = root / "spend.jsonl"
    monkeypatch.setenv(oneshot_mod.LEDGER_ENV, str(ledger))
    summary = await score_prompt_run("r1", task=load_task(TASK), judges=[_priced_judge()])
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 3 and sum(r["cost_usd"] for r in rows) == pytest.approx(1.5)
    assert summary.cases[0].judge_cost_usd == pytest.approx(0.5)


async def test_the_limit_stops_judging_and_writes_no_summary(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _generate()
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _Client.calls = 0
    ledger = root / "spend.jsonl"
    ledger.write_text(json.dumps({"cost_usd": 1.6}) + "\n")
    monkeypatch.setenv(oneshot_mod.LEDGER_ENV, str(ledger))
    monkeypatch.setenv(oneshot_mod.LIMIT_ENV, "2")
    with pytest.raises(SpendLimitReached, match="no summary written"):
        await score_prompt_run("r1", task=load_task(TASK), judges=[_priced_judge()], concurrency=1)
    assert _Client.calls == 1, "one call took the ledger past $2; nothing after it was sent"
    assert not summary_path("r1").exists()


async def test_scoring_a_coding_run_as_a_prompt_run_is_refused(root: Path) -> None:
    with pytest.raises(ValueError, match="not a prompt-task run"):
        await score_prompt_run("nope", task=load_task(TASK), judges=[])


def test_cli_score_routes_a_prompt_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from typer.testing import CliRunner

    from eval_harness.cli import app
    from eval_harness.prompt import judge as judge_mod

    asyncio.run(_generate())
    monkeypatch.setattr(judge_mod, "oneshot", Judges())
    out = CliRunner().invoke(app, ["score", "--run", "r1", "--judge", "judge-a"])
    assert out.exit_code == 0, out.output
    assert "scored 3 attempt(s), 0 not graded; judges=judge-a" in out.output
