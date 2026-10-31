"""Generate one completion per (case, repeat). No Docker and no tools: the call is the attempt."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from eval_harness.adapters.oneshot import OneshotResult, SpendLimitReached, oneshot
from eval_harness.config import ModelConfig
from eval_harness.prompt.cases import PromptCase
from eval_harness.prompt.record import PromptAttempt, load_attempt, save_attempt
from eval_harness.prompt.task import TaskConfig

Generate = Callable[..., Awaitable[OneshotResult]]


@dataclass
class PromptRunResult:
    attempts: list[PromptAttempt] = field(default_factory=list)
    # Why the run stopped before every attempt was made, if it did. The attempts already made
    # are on disk, so the same command resumes it once the reason is dealt with.
    stopped: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def max_tokens_for(case: PromptCase, task: TaskConfig, model: ModelConfig) -> int:
    """The case's own ceiling, then the task's default, then the model's."""
    return case.max_output_tokens or task.defaults.max_output_tokens or model.max_output_tokens


async def run_prompt_cases(
    cases: list[PromptCase],
    model: ModelConfig,
    *,
    task: TaskConfig,
    run_id: str,
    repeats: int = 1,
    concurrency: int = 4,
    retry_errors: bool = False,
    generate: Generate | None = None,
    on_attempt: Callable[[PromptAttempt], Any] | None = None,
) -> PromptRunResult:
    """Make every missing attempt; hand back saved ones marked `reused`.

    A completed attempt is never regenerated: its output is what the judges scored, and a
    second sample under the same (case, repeat) would make the cache lie. An errored one is
    re-run only with `retry_errors`, as for coding runs.
    """
    ask = generate or oneshot
    gate = asyncio.Semaphore(max(1, concurrency))
    out = PromptRunResult()
    stop = asyncio.Event()

    async def one(case: PromptCase, repeat: int) -> PromptAttempt | None:
        saved = load_attempt(run_id, case.case_id, repeat)
        if saved is not None and (saved.status == "completed" or not retry_errors):
            return saved
        async with gate:
            if stop.is_set():
                return None
            started = _now()
            try:
                result = await ask(
                    case.prompt,
                    system=case.system,
                    model=model,
                    max_tokens=max_tokens_for(case, task, model),
                )
            except SpendLimitReached as e:
                # Not an attempt error: the model was never asked. Record nothing, so a
                # resumed run asks it then.
                stop.set()
                out.stopped = str(e)
                return None
            except Exception as e:
                rec = PromptAttempt(
                    case_id=case.case_id,
                    run_id=run_id,
                    model=model.key,
                    repeat=repeat,
                    status="error",
                    prompt_version=case.prompt_version,
                    error=f"{type(e).__name__}: {e}"[:2000],
                    started_at=started,
                    finished_at=_now(),
                )
            else:
                rec = PromptAttempt(
                    case_id=case.case_id,
                    run_id=run_id,
                    model=model.key,
                    repeat=repeat,
                    status="completed",
                    output_text=result.text,
                    usage=result.usage,
                    cost_usd=result.cost_usd,
                    latency_seconds=result.latency_seconds,
                    request_settings=result.request_settings,
                    stop_reason=result.stop_reason,
                    prompt_version=case.prompt_version,
                    started_at=started,
                    finished_at=_now(),
                )
            save_attempt(rec)
            if on_attempt is not None:
                on_attempt(rec)
            return rec

    jobs = [one(c, r) for c in cases for r in range(repeats)]
    for rec in await asyncio.gather(*jobs):
        if rec is not None:
            out.attempts.append(rec)
    out.attempts.sort(key=lambda a: (a.case_id, a.repeat))
    return out
