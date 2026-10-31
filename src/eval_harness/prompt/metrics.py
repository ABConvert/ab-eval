"""Run-level numbers for a prompt run, from its PromptCaseScore rows. Pure Python.

Every mean carries its n and a bootstrap CI, because a prompt benchmark is usually run on a
hundred-odd cases and two models a tenth of a point apart are often not apart at all. Judge
agreement is reported so a reader can tell a quality difference from a judging one, and
repeat self-consistency so they can tell either from sampling noise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean
from typing import TYPE_CHECKING, Any

from eval_harness.prompt.task import Dimension, Rubric
from eval_harness.report.stats import bootstrap_ci, percentile

if TYPE_CHECKING:
    from eval_harness.prompt.score import PromptCaseScore, PromptRunSummary


def _stat(values: list[float], *, ci: bool = True) -> dict[str, Any]:
    return {
        "mean": round(mean(values), 4) if values else None,
        "n": len(values),
        "ci": bootstrap_ci(values) if ci else None,
    }


def quality_of(
    row: PromptCaseScore, rubric: Rubric, quality_metrics: dict[str, float]
) -> float | None:
    """The weighted 0..1 composite for one attempt, over whichever parts it has.

    Rubric dimensions use their judge mean, normalised to 0..1; scorer metrics named in
    task.yaml `quality_metrics` are taken as they are (a bool is 0 or 1). An attempt with
    no gradable part — an error, a score_error — has no quality rather than a quality of 0.
    """
    if row.status != "completed":
        return None
    total = weight = 0.0
    for d in rubric.dimensions:
        if d.name not in row.judge_mean or d.weight <= 0 or (d.is_enum and not d.good):
            continue
        v = row.judge_mean[d.name]
        # An enum's judge mean is already the 0..1 share answering a good label.
        part = v if d.is_enum else (v - d.bounds[0]) / (d.bounds[1] - d.bounds[0])
        total += d.weight * part
        weight += d.weight
    for name, w in quality_metrics.items():
        if name in row.scorer_metrics and w > 0:
            total += w * float(row.scorer_metrics[name])
            weight += w
    return round(total / weight, 4) if weight else None


def cohen_kappa(a: list[Any], b: list[Any]) -> float | None:
    """Unweighted Cohen's κ for labels. None when chance agreement is total (κ undefined)."""
    n = len(a)
    if n < 2:
        return None
    observed: float = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected: float = sum(ca[k] * cb[k] for k in ca) / (n * n)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def weighted_kappa(a: list[int], b: list[int], lo: int, hi: int) -> float | None:
    """Quadratic-weighted κ on an ordinal lo..hi scale: a 4 against a 5 is a near miss."""
    n = len(a)
    if n < 2 or hi <= lo:
        return None
    span = float(hi - lo) ** 2

    def w(x: int, y: int) -> float:
        return 1.0 - (x - y) ** 2 / span

    observed = sum(w(x, y) for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[x] * cb[y] * w(x, y) for x in ca for y in cb) / (n * n)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def agreement(rows: list[PromptCaseScore], rubric: Rubric, judges: list[str]) -> dict[str, Any]:
    """Per judge pair and dimension: κ over attempts both judges scored, and raw agreement."""
    out: dict[str, Any] = {}
    for j1, j2 in combinations(judges, 2):
        per: dict[str, Any] = {}
        for d in rubric.dimensions:
            pairs = [
                (r.judge_scores[j1][d.name], r.judge_scores[j2][d.name])
                for r in rows
                if d.name in r.judge_scores.get(j1, {}) and d.name in r.judge_scores.get(j2, {})
            ]
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            if d.is_enum:
                kappa = cohen_kappa(a, b)
                kind = "cohen"
            else:
                lo, hi = d.bounds
                kappa = weighted_kappa([int(x) for x in a], [int(y) for y in b], lo, hi)
                kind = "quadratic_weighted"
            per[d.name] = {
                "kappa": kappa,
                "kind": kind,
                "n": len(pairs),
                "exact_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4)
                if pairs
                else None,
            }
        out[f"{j1}|{j2}"] = per
    return out


def self_consistency(
    rows: list[PromptCaseScore], rubric: Rubric, quality_metrics: dict[str, float]
) -> dict[str, Any]:
    """How much a case's grade moves between repeats of the same model on the same prompt.

    Mean absolute pairwise difference across repeats, per dimension and for the composite,
    over cases with two or more graded repeats. Zero is a model that answers the same way
    every time; on a 1-5 scale, 1.0 is a full point of noise per resample.
    """
    by_case: dict[str, list[PromptCaseScore]] = defaultdict(list)
    for r in rows:
        if r.status == "completed":
            by_case[r.case_id].append(r)
    multi = {k: v for k, v in by_case.items() if len(v) > 1}
    if not multi:
        return {}

    def spread(values: list[float]) -> float:
        diffs = [abs(x - y) for x, y in combinations(values, 2)]
        return mean(diffs)

    out: dict[str, Any] = {"cases": len(multi)}
    for d in rubric.dimensions:
        per_case = [
            spread([r.judge_mean[d.name] for r in rs])
            for rs in multi.values()
            if sum(d.name in r.judge_mean for r in rs) == len(rs)
        ]
        if per_case:
            out[d.name] = {"mean_abs_diff": round(mean(per_case), 4), "n": len(per_case)}
    q = [
        spread([v for v in vals if v is not None])
        for vals in ([quality_of(r, rubric, quality_metrics) for r in rs] for rs in multi.values())
        if sum(v is not None for v in vals) > 1
    ]
    if q:
        out["quality"] = {"mean_abs_diff": round(mean(q), 4), "n": len(q)}
    return out


def _dimension_block(d: Dimension, rows: list[PromptCaseScore], *, ci: bool) -> dict[str, Any]:
    graded = [r for r in rows if d.name in r.judge_mean]
    block: dict[str, Any] = {"kind": "enum" if d.is_enum else "scale", "scale": d.scale}
    if d.is_enum:
        # The labels judges gave, pooled over judges: what share of answers said what.
        labels: Counter[str] = Counter()
        for r in rows:
            for scores in r.judge_scores.values():
                if d.name in scores:
                    labels[str(scores[d.name])] += 1
        total = sum(labels.values())
        block["distribution"] = {k: round(v / total, 4) for k, v in sorted(labels.items())}
        if d.good:
            block.update(_stat([r.judge_mean[d.name] for r in graded], ci=ci))
            block["good"] = d.good
        else:
            block["n"] = total
    else:
        block.update(_stat([r.judge_mean[d.name] for r in graded], ci=ci))
    disagreement = [r.judge_disagreement[d.name] for r in rows if d.name in r.judge_disagreement]
    block["judge_disagreement_mean"] = round(mean(disagreement), 4) if disagreement else None
    return block


def _scorer_block(rows: list[PromptCaseScore], *, ci: bool) -> dict[str, Any]:
    names = sorted({k for r in rows for k in r.scorer_metrics})
    out: dict[str, Any] = {}
    for name in names:
        values = [float(r.scorer_metrics[name]) for r in rows if name in r.scorer_metrics]
        kind = (
            "bool"
            if all(isinstance(r.scorer_metrics.get(name, False), bool) for r in rows)
            else "number"
        )
        out[name] = {"kind": kind, **_stat(values, ci=ci)}
    return out


def compute(summary: PromptRunSummary, *, ci: bool = True) -> dict[str, Any]:
    rubric = Rubric.model_validate(summary.rubric)
    rows = summary.cases
    qm = summary.quality_metrics
    status = Counter(r.status for r in rows)
    graded = [r for r in rows if r.status == "completed"]
    generated = [r for r in rows if r.status != "error"]
    quality = [q for q in (quality_of(r, rubric, qm) for r in rows) if q is not None]
    cost = sum(r.cost_usd for r in rows)
    case_ids = {r.case_id for r in rows}
    latencies = [r.latency_seconds for r in generated]
    flags: Counter[str] = Counter(f for r in rows for f in r.scorer_flags)

    strata: dict[str, Any] = {}
    labels = sorted(
        {s for r in rows for s in r.strata} | {f"split:{r.split}" for r in rows if r.split}
    )
    for label in labels:
        members = [
            r
            for r in rows
            if (label in r.strata) or (label.startswith("split:") and f"split:{r.split}" == label)
        ]
        strata[label] = {
            "n": len(members),
            "quality": _stat(
                [q for q in (quality_of(r, rubric, qm) for r in members) if q is not None], ci=ci
            ),
            "dimensions": {
                d.name: _stat(
                    [r.judge_mean[d.name] for r in members if d.name in r.judge_mean], ci=False
                )["mean"]
                for d in rubric.dimensions
            },
        }

    return {
        "attempts": len(rows),
        "cases": len(case_ids),
        "repeats": summary.repeats,
        "status": dict(sorted(status.items())),
        "error_rate": round(status.get("error", 0) / len(rows), 4) if rows else None,
        "score_error_rate": round(status.get("score_error", 0) / len(rows), 4) if rows else None,
        "quality": _stat(quality, ci=ci),
        "dimensions": {d.name: _dimension_block(d, graded, ci=ci) for d in rubric.dimensions},
        "scorer": _scorer_block(graded, ci=ci),
        "flags": dict(flags.most_common()),
        "strata": strata,
        "agreement": agreement(graded, rubric, summary.judges),
        "self_consistency": self_consistency(rows, rubric, qm),
        "self_preference": list(summary.self_preference),
        "cost": {
            "total_usd": round(cost, 6),
            "per_case_usd": round(cost / len(case_ids), 6) if case_ids else None,
            "per_attempt_usd": round(cost / len(rows), 6) if rows else None,
            "judge_total_usd": round(sum(r.judge_cost_usd for r in rows), 6),
        },
        "tokens": {
            "input": sum(r.input_tokens for r in rows),
            "output": sum(r.output_tokens for r in rows),
            "output_mean": round(mean(r.output_tokens for r in generated), 1)
            if generated
            else None,
            "truncated": sum(1 for r in generated if r.stop_reason in ("max_tokens", "length")),
        },
        "latency": {
            "p50_s": percentile(latencies, 0.5),
            "p90_s": percentile(latencies, 0.9),
        },
        "quality_per_dollar": (
            round(mean(quality) / (cost / len(case_ids)), 4)
            if quality and cost > 0 and case_ids
            else None
        ),
    }
