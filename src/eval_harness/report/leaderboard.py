"""Rank runs by a documented composite, with token efficiency as the headline.

Every number here is derived from run_metrics; nothing is measured afresh. The composite
is a judgement call, so the components are always carried alongside it and the weights
are declared rather than buried.
"""

from __future__ import annotations

from typing import Any

from eval_harness.report.render import run_metrics
from eval_harness.report.summary import RunSummary

# What the composite rewards, and by how much. Effectiveness leads because a cheap model
# that does not fix the bug is worth nothing; token efficiency is the headline among the
# rest because it is the only cost signal every provider reports.
WEIGHTS: dict[str, float] = {
    "resolve_rate": 0.40,
    "token_efficiency": 0.30,
    "quality": 0.20,
    "speed": 0.10,
}


def token_efficiency(m: dict[str, Any]) -> float | None:
    """Quality-weighted solved cases per million tokens.

    Solving more cases raises it, spending more tokens lowers it, and a solution the judge
    dislikes counts for less than a clean one. Returns None when the run spent no tokens,
    which is how the human-patch reference behaves.
    """
    total = float(m["tokens"]["total"])
    if total <= 0:
        return None
    solved = float(m["overall"]["resolved"])
    quality = m["code_quality_mean"]
    weighted = solved * (float(quality) if quality is not None else 1.0)
    return round(weighted / (total / 1_000_000), 4)


def solves_per_hour(m: dict[str, Any]) -> float | None:
    """Solved cases per hour of agent time, at the median case."""
    median = m["wall_median_s"]
    solved = m["overall"]["resolved"]
    if not median or not solved:
        return None
    return round(3600.0 / float(median), 4)


def _normalise(values: dict[str, float | None]) -> dict[str, float]:
    """Scale to the best run in this comparison; a missing value scores zero."""
    present = [v for v in values.values() if v is not None]
    top = max(present) if present else 0.0
    if top <= 0:
        return {k: 0.0 for k in values}
    return {k: round((v or 0.0) / top, 4) for k, v in values.items()}


def build(summaries: list[RunSummary], weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Rank the given runs. Returns rows plus the weights used, newest metrics recomputed."""
    w = dict(weights or WEIGHTS)
    # The board prints point estimates only, so skip nine bootstraps per run.
    metrics = {s.run_id: run_metrics(s, ci=False) for s in summaries}

    raw = {
        "resolve_rate": {r: m["overall"]["rate"] for r, m in metrics.items()},
        "token_efficiency": {r: token_efficiency(m) for r, m in metrics.items()},
        "quality": {r: m["code_quality_mean"] for r, m in metrics.items()},
        "speed": {r: solves_per_hour(m) for r, m in metrics.items()},
    }
    scaled = {k: _normalise(v) for k, v in raw.items()}

    rows = []
    for s in summaries:
        m = metrics[s.run_id]
        parts = {k: scaled[k][s.run_id] for k in w}
        rows.append(
            {
                "run_id": s.run_id,
                "model": s.model,
                "score": round(sum(parts[k] * w[k] for k in w), 4),
                "parts": parts,
                "resolve_rate": raw["resolve_rate"][s.run_id],
                "resolved": m["overall"]["resolved"],
                "n": m["overall"]["n"],
                "token_efficiency": raw["token_efficiency"][s.run_id],
                "quality": raw["quality"][s.run_id],
                "solves_per_hour": raw["speed"][s.run_id],
                "tokens_total": m["tokens"]["total"],
                "tokens_per_solved": m["tokens_per_solved"],
                "cost_per_task": m["cost_per_task"],
                "cost_total": m["total_cost_usd"],
                "wall_median_s": m["wall_median_s"],
                "wall_p95_s": m["wall_p95_s"],
                "judged": sum(1 for c in s.cases if c.code_quality is not None),
            }
        )
    rows.sort(key=lambda r: (-float(r["score"]), str(r["run_id"])))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return {"weights": w, "rows": rows}
