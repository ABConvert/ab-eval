"""Cases as DeepEval goldens; attempt records as test cases; `score_run` glues it together."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from eval_harness.collect.cases import Case, load_case
from eval_harness.collect.split import load_splits
from eval_harness.config import ModelConfig
from eval_harness.harness.record import AttemptRecord, results_dir
from eval_harness.metrics.code_quality import make_code_quality_metric
from eval_harness.metrics.deterministic import (
    LintCleanMetric,
    NoRegressionMetric,
    PatchSimilarityMetric,
    TestsPassMetric,
)
from eval_harness.metrics.judge import SubscriptionJudge
from eval_harness.paths import cases_dir as default_cases_dir
from eval_harness.paths import splits_path
from eval_harness.report.summary import CaseScore, RunSummary, save_summary


def golden_from_case(case: Case) -> Golden:
    meta = case.model_dump(exclude={"human_patch", "human_test_patch", "task_prompt"})
    return Golden(
        input=case.task_prompt,
        expected_output=case.human_patch,
        additional_metadata=meta,
        multimodal=False,
    )


def goldens_from_cases(cases: list[Case]) -> EvaluationDataset:
    return EvaluationDataset(goldens=[golden_from_case(c) for c in cases])


def test_case_from(record: AttemptRecord, case: Case) -> LLMTestCase:
    after = record.tests_after
    summary = (
        f"[tests] {after.passed}/{after.total} passed, {after.failed} failed"
        if after
        else f"[tests] not run ({record.status})"
    )
    rec = record.model_dump()
    rec["human_patch"] = case.human_patch
    return LLMTestCase(
        name=f"{record.run_id}/{record.case_id}",
        input=case.task_prompt,
        actual_output=f"{record.generated_diff}\n\n{summary}",
        expected_output=case.human_patch,
        metadata={"record": rec, "case_id": case.case_id},
    )


# The run's own files, which live in the same directory as the attempt records and are not
# attempt records. The glob used to be `PRO-*.json`, which excluded them by accident; opening
# the harness up meant dropping one team's case-id prefix, and the two reserved names have to
# be excluded on purpose instead — `run.json` is written before the first case starts, so
# without this every `score` raises before it reads a single attempt.
RUN_FILES = {"run.json", "summary.json"}


def _load_records(run_id: str) -> list[AttemptRecord]:
    out = []
    for path in sorted(results_dir(run_id).glob("*.json")):
        # summary.* covers the original and any second judge's summary written beside it.
        if path.name in RUN_FILES or path.name.startswith("summary."):
            continue
        out.append(AttemptRecord.model_validate_json(path.read_text()))
    return out


def _split_of(case_id: str) -> str:
    if not splits_path().exists():
        return "?"
    splits = load_splits()
    return "holdout" if case_id in set(splits["holdout"]) else "dev"


def _metric_scores(
    results: Any,
) -> dict[str, dict[str, tuple[float | None, str | None]]]:
    """{test name: {metric name: (score, reason)}} from an EvaluationResult."""
    out: dict[str, dict[str, tuple[float | None, str | None]]] = {}
    for tr in getattr(results, "test_results", []) or []:
        per: dict[str, tuple[float | None, str | None]] = {}
        for md in tr.metrics_data or []:
            per[md.name] = (md.score, md.reason or (f"error: {md.error}" if md.error else None))
        out[tr.name] = per
    return out


def score_run(
    run_id: str,
    *,
    judge_cfg: ModelConfig | None,
    use_cache: bool = True,
    cases_dir: Path | None = None,
    max_concurrent: int = 8,
    write_as: str | None = None,
) -> RunSummary:
    """Score a run. `write_as` puts the summary beside summary.json instead of over it.

    That is how a second judge is checked against the first: the dashboard reads
    summary.json, so a second opinion must not replace the one it is being compared with.
    """
    if write_as is not None and (
        Path(write_as).name != write_as
        or not write_as.startswith("summary.")
        or not write_as.endswith(".json")
    ):
        raise ValueError(
            f"write_as must be a summary.*.json file name in the run directory: {write_as!r}"
        )
    cases_dir = cases_dir or default_cases_dir()
    records = _load_records(run_id)
    if not records:
        raise FileNotFoundError(f"no attempt records under results/{run_id}")
    meta_path = results_dir(run_id) / "run.json"
    run_meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cases = {r.case_id: load_case(r.case_id, cases_dir) for r in records}
    test_cases = [test_case_from(r, cases[r.case_id]) for r in records]

    deterministic: list[BaseMetric] = [
        TestsPassMetric(),
        NoRegressionMetric(),
        LintCleanMetric(),
        PatchSimilarityMetric(),
    ]
    common = {
        "async_config": AsyncConfig(run_async=True, max_concurrent=max_concurrent),
        "cache_config": CacheConfig(write_cache=True, use_cache=use_cache),
        "display_config": DisplayConfig(show_indicator=False, print_results=False),
        "error_config": ErrorConfig(ignore_errors=True),
    }
    det = _metric_scores(evaluate(test_cases=test_cases, metrics=deterministic, **common))  # type: ignore[arg-type]

    quality: dict[str, tuple[float | None, str | None]] = {}
    judge_name: str | None = None
    if judge_cfg is not None:
        judge = SubscriptionJudge(judge_cfg)
        judge_name = judge.get_model_name()
        resolved_cases = [tc for tc, r in zip(test_cases, records, strict=True) if r.resolved]
        if resolved_cases:
            quality_metrics: list[BaseMetric] = [make_code_quality_metric(judge)]
            res = evaluate(
                test_cases=resolved_cases,
                metrics=quality_metrics,
                async_config=AsyncConfig(run_async=False),
                cache_config=CacheConfig(write_cache=True, use_cache=use_cache),
                display_config=DisplayConfig(show_indicator=False, print_results=False),
                error_config=ErrorConfig(ignore_errors=True),
            )
            for name, per in _metric_scores(res).items():
                # DeepEval suffixes G-Eval metric names with " [GEval]".
                quality[name] = next(
                    (v for k, v in per.items() if k.startswith("Code quality")), (None, None)
                )

    scored: list[CaseScore] = []
    for r, tc in zip(records, test_cases, strict=True):
        case = cases[r.case_id]
        per = det.get(tc.name or "", {})
        q = quality.get(tc.name or "", (None, None))
        scored.append(
            CaseScore(
                case_id=r.case_id,
                kind=case.kind,
                tier=case.tier,
                author_kind=case.author_kind,
                spec_level=case.spec_level_heuristic,
                split=_split_of(r.case_id),
                status=r.status,
                resolved=r.resolved,
                tests_pass=float(per.get("Tests pass", (0.0, None))[0] or 0.0),
                no_regression=per.get("No regression", (None, None))[0] if r.resolved else None,
                lint_clean=per.get("Lint clean", (None, None))[0] if r.resolved else None,
                patch_similarity=per.get("Patch similarity", (None, None))[0],
                code_quality=q[0],
                code_quality_reason=q[1],
                cost_usd=r.cost_usd,
                input_tokens=r.usage.input_tokens,
                output_tokens=r.usage.output_tokens,
                cache_read_tokens=r.usage.cache_read_tokens,
                cache_write_tokens=r.usage.cache_write_tokens,
                wall_clock_seconds=r.wall_clock_seconds,
                agent_wall_clock_seconds=r.agent_wall_clock_seconds,
                turns=r.turns,
                tool_calls=r.tool_calls,
                cap_hit=r.cap_hit,
                touched_test_files=bool(r.touched_test_files),
                host_exec_items=len(r.request_settings.get("host_exec_items") or []),
                files_touched=len(r.files_touched),
                diff_lines=sum(1 for line in r.generated_diff.splitlines() if line[:1] in "+-"),
            )
        )
    from eval_harness.collect.difficulty import classify

    summary = RunSummary(
        run_id=run_id,
        model=str(run_meta.get("model") or (records[0].model if records else "?")),
        model_config_snapshot=dict(run_meta.get("model_config_snapshot") or {}),
        caps=dict(run_meta.get("caps") or {}),
        harness_git_sha=str(run_meta.get("harness_git_sha") or "?"),
        judge=judge_name,
        judge_config=judge_cfg.model_dump() if judge_cfg else None,
        case_difficulties={cid: classify(case) for cid, case in cases.items()},
        scored_at=datetime.now(UTC).isoformat(timespec="seconds"),
        cases=scored,
    )
    # Metric errors are ignored above, so a judge stopped by the spend limit would otherwise
    # leave cases unscored in a summary that reads as finished.
    from eval_harness.adapters.oneshot import LIMIT_ENV, SpendLimitReached, spent_so_far

    limit = os.environ.get(LIMIT_ENV)
    if limit and spent_so_far() >= float(limit):
        raise SpendLimitReached(
            f"{run_id}: the judge spend limit of ${float(limit):.2f} was reached while scoring; "
            "no summary written"
        )
    if write_as is None:
        save_summary(summary)
    else:
        (results_dir(run_id) / write_as).write_text(
            json.dumps(summary.model_dump(), indent=2) + "\n"
        )
    return summary
