"""Score a prompt run: scorer per attempt, every judge per attempt, one summary per run.

The summary is written to results/<run>/prompt-summary.json, not summary.json: every reader
of summary.json expects a coding RunSummary, and a prompt run's has another shape.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from eval_harness.adapters.oneshot import SpendLimitReached
from eval_harness.config import ModelConfig
from eval_harness.harness.record import results_dir
from eval_harness.prompt.cases import PromptCase, load_prompt_case
from eval_harness.prompt.judge import Generate, Judgement, judge_attempt
from eval_harness.prompt.record import PromptAttempt, load_attempts, load_run_meta
from eval_harness.prompt.scorer import ScorerError, ScorerOutput, run_scorer
from eval_harness.prompt.task import Rubric, TaskConfig

SUMMARY_FILE = "prompt-summary.json"


class PromptCaseScore(BaseModel):
    case_id: str
    repeat: int
    split: str | None
    strata: list[str] = Field(default_factory=list)
    # completed: generated and graded. error: the model call failed, nothing to grade.
    # score_error: generated, but the scorer failed or every judge did — so its grade is
    # unknown, which is not the same as bad and is never averaged in as zero.
    status: Literal["completed", "error", "score_error"]
    error: str | None = None
    scorer_metrics: dict[str, float | bool] = Field(default_factory=dict)
    scorer_flags: list[str] = Field(default_factory=list)
    scorer_error: str | None = None
    judge_scores: dict[str, dict[str, int | str]] = Field(default_factory=dict)
    judge_reasons: dict[str, dict[str, str]] = Field(default_factory=dict)
    judge_errors: dict[str, str] = Field(default_factory=dict)
    # Per dimension, over the judges that answered: the mean for a 1-5 scale, the share of
    # judges answering a `good` label for an enum that declares one.
    judge_mean: dict[str, float] = Field(default_factory=dict)
    # Spread between judges: max - min on a scale, 1.0 if any two disagree on an enum.
    judge_disagreement: dict[str, float] = Field(default_factory=dict)
    cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    stop_reason: str | None = None


class PromptRunSummary(BaseModel):
    run_id: str
    kind: Literal["prompt"] = "prompt"
    task: str
    model: str
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = None
    repeats: int = 1
    rubric: dict[str, Any] = Field(default_factory=dict)
    rubric_hash: str = ""
    judges: list[str] = Field(default_factory=list)
    judge_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Judges that are also the model under test. Their scores are kept and flagged, as the
    # code-quality judge guidance asks, so a reader can drop them.
    self_preference: list[str] = Field(default_factory=list)
    scorer: list[str] | None = None
    # task.yaml's weights for scorer metrics in the quality composite, as they were when scored.
    quality_metrics: dict[str, float] = Field(default_factory=dict)
    harness_git_sha: str = "?"
    scored_at: str = ""
    cases: list[PromptCaseScore]
    metrics: dict[str, Any] = Field(default_factory=dict)


def summary_path(run_id: str) -> Path:
    return results_dir(run_id) / SUMMARY_FILE


def load_prompt_summary(run_id: str) -> PromptRunSummary:
    return PromptRunSummary.model_validate_json(summary_path(run_id).read_text())


def save_prompt_summary(s: PromptRunSummary) -> Path:
    path = summary_path(s.run_id)
    path.write_text(json.dumps(s.model_dump(), indent=2, ensure_ascii=False) + "\n")
    return path


def self_preferring(
    run_model: dict[str, Any], run_key: str, judges: list[ModelConfig]
) -> list[str]:
    model_id = str(run_model.get("model") or "")
    return [j.key for j in judges if j.key == run_key or (model_id and j.model == model_id)]


def combine(
    rubric: Rubric, judgements: list[Judgement]
) -> tuple[dict[str, float], dict[str, float]]:
    """(judge_mean, judge_disagreement) per dimension, over judgements without an error."""
    ok = [j for j in judgements if j.error is None]
    mean: dict[str, float] = {}
    spread: dict[str, float] = {}
    for d in rubric.dimensions:
        values = [j.scores[d.name].value for j in ok if d.name in j.scores]
        if not values:
            continue
        if d.is_enum:
            if d.good:
                mean[d.name] = round(sum(v in d.good for v in values) / len(values), 4)
            spread[d.name] = 1.0 if len(set(values)) > 1 else 0.0
        else:
            nums = [float(v) for v in values]
            mean[d.name] = round(sum(nums) / len(nums), 4)
            spread[d.name] = max(nums) - min(nums)
    return mean, spread


async def score_prompt_run(
    run_id: str,
    *,
    task: TaskConfig,
    judges: list[ModelConfig],
    use_cache: bool = True,
    concurrency: int = 4,
    generate: Generate | None = None,
) -> PromptRunSummary:
    meta = load_run_meta(run_id)
    if meta is None or meta.kind != "prompt":
        raise ValueError(f"{run_id} is not a prompt-task run (no run.json with kind: prompt)")
    if meta.task != task.task:
        raise ValueError(f"{run_id} ran task {meta.task!r}, not {task.task!r}")
    attempts = load_attempts(run_id)
    if not attempts:
        raise FileNotFoundError(f"no attempts under results/{run_id}/attempts")
    rubric = task.load_rubric()
    cases: dict[str, PromptCase] = {
        cid: load_prompt_case(task.task, cid) for cid in {a.case_id for a in attempts}
    }
    gate = asyncio.Semaphore(max(1, concurrency))

    async def scorer_for(a: PromptAttempt) -> ScorerOutput | ScorerError | None:
        if task.scorer is None or a.status != "completed":
            return None
        cfg = task.scorer
        async with gate:
            try:
                return await asyncio.to_thread(
                    run_scorer,
                    cfg,
                    case=cases[a.case_id],
                    output_text=a.output_text,
                    cwd=task.directory,
                )
            except ScorerError as e:
                return e

    async def judged(a: PromptAttempt, judge: ModelConfig) -> Judgement:
        async with gate:
            return await judge_attempt(
                judge, rubric, cases[a.case_id], a, use_cache=use_cache, generate=generate
            )

    graded = [a for a in attempts if a.status == "completed"]
    scorer_results = await asyncio.gather(*(scorer_for(a) for a in attempts))
    # return_exceptions, so that one judge stopped by the limit does not leave its siblings'
    # refusals unretrieved; judge_attempt turns every other failure into a recorded error.
    outcomes = await asyncio.gather(
        *(judged(a, j) for a in graded for j in judges), return_exceptions=True
    )
    for o in outcomes:
        if isinstance(o, SpendLimitReached):
            raise SpendLimitReached(
                f"{run_id}: the spend limit was reached while judging; no summary written ({o})"
            ) from o
        if isinstance(o, BaseException):
            raise o
    by_attempt: dict[tuple[str, int], list[Judgement]] = {}
    for j in outcomes:
        assert isinstance(j, Judgement)
        by_attempt.setdefault((j.case_id, j.repeat), []).append(j)

    rows: list[PromptCaseScore] = []
    for a, sr in zip(attempts, scorer_results, strict=True):
        case = cases[a.case_id]
        row = PromptCaseScore(
            case_id=a.case_id,
            repeat=a.repeat,
            split=case.split,
            strata=case.strata,
            status="completed" if a.status == "completed" else "error",
            error=a.error,
            cost_usd=a.cost_usd,
            input_tokens=a.usage.input_tokens + a.usage.cache_read_tokens,
            output_tokens=a.usage.output_tokens,
            latency_seconds=a.latency_seconds,
            stop_reason=a.stop_reason,
        )
        if isinstance(sr, ScorerError):
            row.status = "score_error"
            row.scorer_error = str(sr)
        elif sr is not None:
            row.scorer_metrics = sr.metrics
            row.scorer_flags = sr.flags
        js = by_attempt.get((a.case_id, a.repeat), [])
        for j in js:
            if j.error:
                row.judge_errors[j.judge] = j.error
            else:
                row.judge_scores[j.judge] = {k: v.value for k, v in j.scores.items()}
                row.judge_reasons[j.judge] = {k: v.reason for k, v in j.scores.items()}
            row.judge_cost_usd = round(row.judge_cost_usd + j.cost_usd, 6)
        row.judge_mean, row.judge_disagreement = combine(rubric, js)
        if a.status == "completed" and judges and not row.judge_scores:
            row.status = "score_error"
        rows.append(row)

    summary = PromptRunSummary(
        run_id=run_id,
        task=task.task,
        model=meta.model,
        model_config_snapshot=meta.model_config_snapshot,
        dataset=meta.dataset,
        repeats=meta.repeats,
        rubric=rubric.model_dump(),
        rubric_hash=rubric.hash(),
        judges=[j.key for j in judges],
        judge_configs={j.key: j.model_dump() for j in judges},
        self_preference=self_preferring(meta.model_config_snapshot, meta.model, judges),
        scorer=list(task.scorer.command) if task.scorer else None,
        harness_git_sha=meta.harness_git_sha,
        scored_at=datetime.now(UTC).isoformat(timespec="seconds"),
        cases=rows,
        quality_metrics=dict(task.quality_metrics),
    )
    from eval_harness.prompt.metrics import compute

    summary.metrics = compute(summary)
    save_prompt_summary(summary)
    return summary
