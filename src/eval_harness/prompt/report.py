"""Compare, leaderboard and export for prompt runs. Every number comes from the summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_harness.paths import results_root
from eval_harness.prompt.metrics import quality_of
from eval_harness.prompt.score import PromptCaseScore, PromptRunSummary
from eval_harness.prompt.task import Rubric
from eval_harness.report.stats import paired_bootstrap_diff

# --------------------------------------------------------------------------------- compare


def _paired(
    a: PromptRunSummary, b: PromptRunSummary
) -> list[tuple[PromptCaseScore, PromptCaseScore]]:
    """Attempts graded in both runs, matched on (case_id, repeat)."""
    left = {(r.case_id, r.repeat): r for r in a.cases if r.status == "completed"}
    return [
        (left[(r.case_id, r.repeat)], r)
        for r in b.cases
        if r.status == "completed" and (r.case_id, r.repeat) in left
    ]


def _diff(xs: list[float], ys: list[float]) -> dict[str, Any]:
    d, ci = paired_bootstrap_diff(xs, ys)
    return {
        "n": len(xs),
        "a_mean": round(sum(xs) / len(xs), 4) if xs else None,
        "b_mean": round(sum(ys) / len(ys), 4) if ys else None,
        "diff": d,
        "ci": ci,
        # Only a CI that excludes zero says the runs differ; the point estimate alone does not.
        "significant": bool(ci and (ci[0] > 0 or ci[1] < 0)),
    }


def compare(summaries: list[PromptRunSummary]) -> dict[str, Any]:
    """Each run against the first: paired-bootstrap differences, positive favours the other."""
    if len(summaries) < 2:
        raise ValueError("give at least two prompt runs")
    base = summaries[0]
    tasks = {s.task for s in summaries}
    if len(tasks) > 1:
        raise ValueError(f"runs of different tasks cannot be paired: {sorted(tasks)}")
    hashes = {s.rubric_hash for s in summaries}
    rubric = Rubric.model_validate(base.rubric)
    out: dict[str, Any] = {
        "task": base.task,
        "baseline": base.run_id,
        "rubric_mismatch": len(hashes) > 1,
        "against": [],
    }
    for other in summaries[1:]:
        pairs = _paired(base, other)
        dims: dict[str, Any] = {}
        for d in rubric.dimensions:
            both = [(x, y) for x, y in pairs if d.name in x.judge_mean and d.name in y.judge_mean]
            if both:
                dims[d.name] = _diff(
                    [x.judge_mean[d.name] for x, _ in both],
                    [y.judge_mean[d.name] for _, y in both],
                )
        scorer: dict[str, Any] = {}
        names = {k for x, _ in pairs for k in x.scorer_metrics}
        for name in sorted(names):
            both = [
                (x, y) for x, y in pairs if name in x.scorer_metrics and name in y.scorer_metrics
            ]
            if both:
                scorer[name] = _diff(
                    [float(x.scorer_metrics[name]) for x, _ in both],
                    [float(y.scorer_metrics[name]) for _, y in both],
                )
        qa = [quality_of(x, rubric, base.quality_metrics) for x, _ in pairs]
        qb = [quality_of(y, rubric, other.quality_metrics) for _, y in pairs]
        qpairs = [(x, y) for x, y in zip(qa, qb, strict=True) if x is not None and y is not None]
        out["against"].append(
            {
                "run_id": other.run_id,
                "model": other.model,
                "paired": len(pairs),
                "quality": _diff([x for x, _ in qpairs], [y for _, y in qpairs])
                if qpairs
                else None,
                "dimensions": dims,
                "scorer": scorer,
                "cost_per_case_usd": (other.metrics.get("cost") or {}).get("per_case_usd"),
                "latency_p50_s": (other.metrics.get("latency") or {}).get("p50_s"),
            }
        )
    out["baseline_model"] = base.model
    out["baseline_cost_per_case_usd"] = (base.metrics.get("cost") or {}).get("per_case_usd")
    out["baseline_latency_p50_s"] = (base.metrics.get("latency") or {}).get("p50_s")
    return out


def compare_markdown(cmp: dict[str, Any]) -> str:
    lines = [
        f"# Prompt-task comparison: {cmp['task']}",
        "",
        f"Baseline: `{cmp['baseline']}` ({cmp['baseline_model']}). Differences are other minus "
        "baseline over attempts graded in both, with 95% paired-bootstrap intervals; "
        "**bold** where the interval excludes zero.",
    ]
    if cmp["rubric_mismatch"]:
        lines += [
            "",
            "> The runs were judged with different rubrics; dimension scores may not compare.",
        ]
    for o in cmp["against"]:
        lines += [
            "",
            f"## `{o['run_id']}` ({o['model']}) — {o['paired']} paired attempts",
            "",
            "| measure | baseline | this run | diff | 95% CI |",
            "|---|---:|---:|---:|---|",
        ]
        rows: list[tuple[str, dict[str, Any] | None]] = [("quality (0..1)", o["quality"])]
        rows += [(f"rubric: {k}", v) for k, v in o["dimensions"].items()]
        rows += [(f"scorer: {k}", v) for k, v in o["scorer"].items()]
        for name, r in rows:
            if not r:
                continue
            diff = f"{r['diff']:+.3f}"
            if r["significant"]:
                diff = f"**{diff}**"
            ci = f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]" if r["ci"] else "-"
            lines.append(f"| {name} | {r['a_mean']:.3f} | {r['b_mean']:.3f} | {diff} | {ci} |")
        lines += [
            "",
            f"Cost per case: ${_money(cmp['baseline_cost_per_case_usd'])} → "
            f"${_money(o['cost_per_case_usd'])}; p50 latency: "
            f"{_secs(cmp['baseline_latency_p50_s'])} → {_secs(o['latency_p50_s'])}.",
        ]
    return "\n".join(lines) + "\n"


def _money(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def _secs(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}s"


def write_comparison(summaries: list[PromptRunSummary]) -> tuple[Path, Path]:
    cmp = compare(summaries)
    stem = "compare-prompt-" + "-vs-".join(s.run_id for s in summaries)
    root = results_root()
    root.mkdir(parents=True, exist_ok=True)
    js, md = root / f"{stem}.json", root / f"{stem}.md"
    js.write_text(json.dumps(cmp, indent=2) + "\n")
    md.write_text(compare_markdown(cmp))
    return js, md


# ------------------------------------------------------------------------------ leaderboard


def leaderboard(summaries: list[PromptRunSummary]) -> list[dict[str, Any]]:
    """One board per (task, dataset): a model is ranked only against runs of the same cases.

    Ranked by mean quality; cost per case, latency and quality per dollar ride alongside so
    a cheaper model that is not measurably worse is visible as such.
    """
    groups: dict[tuple[str, str], list[PromptRunSummary]] = defaultdict(list)
    for s in summaries:
        groups[(s.task, s.dataset or "(ad hoc)")].append(s)
    boards = []
    for (task, dataset), runs in sorted(groups.items()):
        rows = []
        for s in runs:
            m = s.metrics
            q = m.get("quality") or {}
            rows.append(
                {
                    "run_id": s.run_id,
                    "model": s.model,
                    "quality": q.get("mean"),
                    "quality_ci": q.get("ci"),
                    "n": q.get("n", 0),
                    "attempts": m.get("attempts", len(s.cases)),
                    "error_rate": m.get("error_rate"),
                    "score_error_rate": m.get("score_error_rate"),
                    "cost_per_case_usd": (m.get("cost") or {}).get("per_case_usd"),
                    "latency_p50_s": (m.get("latency") or {}).get("p50_s"),
                    "latency_p90_s": (m.get("latency") or {}).get("p90_s"),
                    "quality_per_dollar": m.get("quality_per_dollar"),
                    "self_preference": list(s.self_preference),
                    "scored_at": s.scored_at,
                }
            )
        rows.sort(key=lambda r: (-(r["quality"] if r["quality"] is not None else -1), r["run_id"]))
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        rubrics = {s.rubric_hash for s in runs}
        boards.append(
            {
                "task": task,
                "dataset": dataset,
                "rows": rows,
                "rubric_mismatch": len(rubrics) > 1,
            }
        )
    return boards


# ----------------------------------------------------------------------------------- export


def summary_json(s: PromptRunSummary) -> dict[str, Any]:
    """The run summary with no case-level text, for a product's own admin page.

    Kept: numbers, case ids (opaque, produced by the exporter), splits and strata labels.
    Dropped: outputs, judge reasons, error messages and scorer flags per case, any of which
    can quote a prompt or an output. The rubric goes as dimension names and scales only.
    """
    rubric = Rubric.model_validate(s.rubric)
    return {
        "format": "ab-eval/prompt-summary@1",
        "run_id": s.run_id,
        "task": s.task,
        "model": s.model,
        "model_id": s.model_config_snapshot.get("model"),
        "provider": s.model_config_snapshot.get("provider"),
        "dataset": s.dataset,
        "repeats": s.repeats,
        "rubric_hash": s.rubric_hash,
        "rubric": [
            {"name": d.name, "scale": d.scale, "good": d.good, "weight": d.weight}
            for d in rubric.dimensions
        ],
        "judges": [
            {"key": k, "model": (s.judge_configs.get(k) or {}).get("model")} for k in s.judges
        ],
        "self_preference": s.self_preference,
        "harness_git_sha": s.harness_git_sha,
        "scored_at": s.scored_at,
        "metrics": s.metrics,
        "cases": [
            {
                "case_id": r.case_id,
                "repeat": r.repeat,
                "split": r.split,
                "strata": r.strata,
                "status": r.status,
                "quality": quality_of(r, rubric, s.quality_metrics),
                "judge_mean": r.judge_mean,
                "judge_scores": r.judge_scores,
                "scorer_metrics": r.scorer_metrics,
                "cost_usd": r.cost_usd,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_seconds": r.latency_seconds,
            }
            for r in s.cases
        ],
    }
