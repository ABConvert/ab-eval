"""The prompt runner: one completion per (case, repeat), recorded, resumable, no Docker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from eval_harness import paths
from eval_harness.adapters.oneshot import OneshotResult, SpendLimitReached
from eval_harness.harness.record import RunMeta
from eval_harness.prompt.cases import load_prompt_cases
from eval_harness.prompt.record import attempt_path, load_attempts
from eval_harness.prompt.runner import max_tokens_for, run_prompt_cases
from eval_harness.prompt.task import load_task
from tests.prompt_helpers import TASK, clear_spend_env, model, result, setup_root


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    clear_spend_env(monkeypatch)
    paths.reset_cache()
    setup_root(tmp_path)
    return tmp_path


class Recorder:
    """A fake one-shot that records every call and answers from a script."""

    def __init__(self, reply: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reply = reply or (lambda prompt, n: result(f"summary {n} of interval run"))

    async def __call__(self, prompt: str, **kw: Any) -> OneshotResult:
        self.calls.append({"prompt": prompt, **kw})
        out = self.reply(prompt, len(self.calls))
        if isinstance(out, BaseException):
            raise out
        assert isinstance(out, OneshotResult)
        return out


async def _run(gen: Recorder, **kw: Any) -> Any:
    task = load_task(TASK)
    return await run_prompt_cases(
        load_prompt_cases(TASK), model("cand"), task=task, run_id="r1", generate=gen, **kw
    )


async def test_each_case_and_repeat_is_one_recorded_attempt(root: Path) -> None:
    gen = Recorder()
    out = await _run(gen, repeats=2)
    assert len(out.attempts) == 6 and len(gen.calls) == 6
    assert attempt_path("r1", "demo-sum-0001", 1).exists()
    rec = load_attempts("r1")[0]
    assert rec.status == "completed" and rec.output_text.startswith("summary")
    assert rec.usage.output_tokens == 100 and rec.cost_usd == 0.01
    assert rec.latency_seconds == 0.5 and rec.request_settings == {"provider": "fake"}
    assert rec.prompt_version == "demo-v1"


async def test_the_model_sees_the_prompt_and_never_the_grading_context(root: Path) -> None:
    gen = Recorder()
    await _run(gen)
    case = load_prompt_cases(TASK)[0]
    call = gen.calls[0]
    assert call["system"] == case.system
    assert call["prompt"] == case.prompt
    assert '"tier": "synthetic"' not in call["prompt"], "labels are for graders only"


async def test_a_completed_attempt_is_reused_never_regenerated(root: Path) -> None:
    await _run(Recorder())
    again = Recorder()
    out = await _run(again)
    assert again.calls == []
    assert all(a.reused for a in out.attempts), "a saved result must say it is one"


async def test_errors_are_recorded_and_retried_only_when_asked(root: Path) -> None:
    await _run(Recorder(lambda p, n: RuntimeError("rate limited")))
    recs = load_attempts("r1")
    assert {r.status for r in recs} == {"error"} and "rate limited" in (recs[0].error or "")
    untouched = Recorder()
    await _run(untouched)
    assert untouched.calls == []
    retried = Recorder()
    await _run(retried, retry_errors=True)
    assert len(retried.calls) == 3
    assert {r.status for r in load_attempts("r1")} == {"completed"}


async def test_the_spend_limit_stops_the_run_and_records_nothing_unasked(root: Path) -> None:
    def reply(prompt: str, n: int) -> object:
        return SpendLimitReached("ledger at $5") if n >= 2 else result("ok")

    out = await _run(Recorder(reply), concurrency=1)
    assert out.stopped and "$5" in out.stopped
    assert len(load_attempts("r1")) == 1, "a refused call is not an attempt"


def test_max_tokens_prefers_case_then_task_then_model(root: Path) -> None:
    task = load_task(TASK)
    case = load_prompt_cases(TASK)[0]
    cand = model("cand")
    assert max_tokens_for(case, task, cand) == 3000
    case.max_output_tokens = None
    assert max_tokens_for(case, task, cand) == 3000  # task default
    task.defaults.max_output_tokens = None
    assert max_tokens_for(case, task, cand) == 4000  # the model's own


def test_a_run_json_written_before_prompt_tasks_is_a_code_run() -> None:
    old = {
        "run_id": "x",
        "model": "m",
        "model_config_snapshot": {},
        "caps": {},
        "concurrency": 1,
        "harness_git_sha": "abc",
        "image": "node20",
        "cases": [],
        "started_at": "2026-01-01",
    }
    meta = RunMeta.model_validate(old)
    assert meta.kind == "code" and meta.task is None and meta.repeats == 1


def test_cli_run_task_records_a_prompt_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eval_harness.cli import app
    from eval_harness.prompt import runner

    gen = Recorder()
    monkeypatch.setattr(runner, "oneshot", gen)
    out = CliRunner().invoke(
        app,
        [
            "run",
            "--task",
            TASK,
            "--model",
            "cand",
            "--dataset",
            "dev",
            "--repeats",
            "2",
            "--run-id",
            "cli-run",
            "--no-score",
        ],
    )
    assert out.exit_code == 0, out.output
    meta = json.loads((root / "results" / "cli-run" / "run.json").read_text())
    assert meta["kind"] == "prompt" and meta["task"] == TASK and meta["repeats"] == 2
    assert meta["cases"] == ["demo-sum-0001", "demo-sum-0002"] and meta["dataset"] == "dev"
    assert len(gen.calls) == 4
    assert "4 of 4 attempt(s), 0 error(s)" in out.output


def test_cli_run_task_with_a_limit_gets_a_monthly_ledger(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval_harness.cli import app
    from eval_harness.prompt import runner

    monkeypatch.setattr(runner, "oneshot", Recorder())
    monkeypatch.setenv("ABEVAL_SPEND_LIMIT_USD", "2")
    out = CliRunner().invoke(
        app, ["run", "--task", TASK, "--model", "cand", "--run-id", "lim", "--no-score"]
    )
    assert out.exit_code == 0, out.output
    assert "_spend" in out.output, "a limit with nowhere to count would never trip"
