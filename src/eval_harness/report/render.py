# ruff: noqa: E501  (Markdown rows are clearer on one line)
"""Markdown and JSON views of one run and of a head-to-head comparison."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from eval_harness.paths import results_root
from eval_harness.report.stats import bootstrap_ci, paired_bootstrap_diff, percentile
from eval_harness.report.summary import CaseScore, RunSummary

MIN_CASES_FOR_RATE = 5


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _ci(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f" [{100 * ci[0]:.0f}-{100 * ci[1]:.0f}]"


def _money(x: float | None) -> str:
    return "-" if x is None else f"${x:.2f}"


def _num(x: float | None, unit: str = "") -> str:
    return "-" if x is None else f"{x:.0f}{unit}"


def resolve_stats(cases: list[CaseScore], *, ci: bool = True) -> dict[str, Any]:
    attempted = [c for c in cases if c.status != "invalid"]
    flags = [1.0 if c.resolved else 0.0 for c in attempted]
    return {
        "n": len(attempted),
        "resolved": int(sum(flags)),
        "rate": round(mean(flags), 4) if flags else None,
        # Ten thousand resamples per slice, nine slices per run. A report wants them; a ranking
        # table that never prints an interval should not pay for them.
        "ci95": bootstrap_ci(flags) if ci and len(flags) >= 2 else None,
        "invalid": sum(c.status == "invalid" for c in cases),
        "errors": sum(c.status == "error" for c in cases),
    }


@lru_cache(maxsize=512)
def difficulty_of(case_id: str) -> str:
    """Rated from the case file, so it applies to runs scored before ratings existed."""
    from eval_harness.collect.cases import load_case
    from eval_harness.collect.difficulty import classify

    try:
        return classify(load_case(case_id))
    except Exception:
        return "unknown"


def case_tokens(c: CaseScore) -> int:
    """Every token the provider reported for one attempt, cached context included."""
    return c.input_tokens + c.output_tokens + c.cache_read_tokens + c.cache_write_tokens


_METRICS_CACHE: dict[tuple[str, str, int, bool], dict[str, Any]] = {}


def run_metrics(s: RunSummary, *, ci: bool = True) -> dict[str, Any]:
    """Memoised: the dashboard asks for the same run's metrics on three different pages."""
    key = (s.run_id, s.scored_at, len(s.cases), ci)
    hit = _METRICS_CACHE.get(key)
    if hit is not None:
        return hit
    out = _run_metrics(s, ci=ci)
    _METRICS_CACHE[key] = out
    return out


def _run_metrics(s: RunSummary, *, ci: bool = True) -> dict[str, Any]:
    """`ci=False` skips the bootstraps, for callers that only read the point estimates."""
    cases = s.cases
    attempted = [c for c in cases if c.status != "invalid"]
    solved = [c for c in cases if c.resolved]
    by_kind = {
        k: resolve_stats([c for c in cases if c.kind == k], ci=ci) for k in ("bug_fix", "feature")
    }
    by_author = {
        k: resolve_stats([c for c in cases if c.author_kind == k], ci=ci)
        for k in sorted({c.author_kind for c in cases})
    }
    by_spec = {
        k: resolve_stats([c for c in cases if c.spec_level == k], ci=ci)
        for k in sorted({c.spec_level for c in cases})
    }
    by_split = {
        k: resolve_stats([c for c in cases if c.split == k], ci=ci) for k in ("dev", "holdout")
    }
    by_difficulty = {
        k: resolve_stats(
            [
                c
                for c in cases
                if (
                    s.case_difficulties[c.case_id]
                    if c.case_id in s.case_difficulties
                    else difficulty_of(c.case_id)
                )
                == k
            ],
            ci=ci,
        )
        for k in ("easy", "medium", "hard")
    }
    cost_solved = [c.cost_usd for c in solved]
    cost_all = [c.cost_usd for c in attempted]
    walls = [
        c.agent_wall_clock_seconds for c in attempted if c.agent_wall_clock_seconds is not None
    ]
    tools = [float(c.tool_calls) for c in attempted]
    total_cost = sum(cost_all)
    quality = [c.code_quality for c in solved if c.code_quality is not None]
    tokens = {
        "input": sum(c.input_tokens for c in attempted),
        "output": sum(c.output_tokens for c in attempted),
        "cache_read": sum(c.cache_read_tokens for c in attempted),
        "cache_write": sum(c.cache_write_tokens for c in attempted),
    }
    tokens["total"] = sum(tokens.values())
    per_case_tokens = [float(case_tokens(c)) for c in attempted]
    per_case_output = [float(c.output_tokens) for c in attempted]
    return {
        "overall": resolve_stats(cases, ci=ci),
        "by_kind": by_kind,
        "by_author_kind": by_author,
        "by_spec_level": by_spec,
        "by_split": by_split,
        "by_difficulty": by_difficulty,
        "cost_per_solved_case": round(total_cost / len(solved), 4) if solved else None,
        # Per task (every attempt, solved or not): what one case costs you to try.
        "cost_per_task": round(total_cost / len(attempted), 4) if attempted else None,
        "tokens_per_task": round(tokens["total"] / len(attempted)) if attempted else None,
        "cost_solved_median": percentile(cost_solved, 0.5),
        "cost_solved_p90": percentile(cost_solved, 0.9),
        "cost_attempted_median": percentile(cost_all, 0.5),
        "cost_attempted_p90": percentile(cost_all, 0.9),
        # p95 is the tail a scheduler has to budget for: with 14 cases it is effectively
        # the worst case, so read it as "how bad does one case get", not as a stable stat.
        "cost_attempted_p95": percentile(cost_all, 0.95),
        "total_cost_usd": round(total_cost, 4),
        "wall_median_s": percentile(walls, 0.5),
        "wall_p90_s": percentile(walls, 0.9),
        "wall_p95_s": percentile(walls, 0.95),
        "tool_calls_median": percentile(tools, 0.5),
        "tool_calls_p90": percentile(tools, 0.9),
        "tool_calls_p95": percentile(tools, 0.95),
        "no_regression_rate": (
            round(mean(c.no_regression or 0.0 for c in solved), 4) if solved else None
        ),
        "lint_clean_rate": round(mean(c.lint_clean or 0.0 for c in solved), 4) if solved else None,
        "code_quality_mean": round(mean(quality), 4) if quality else None,
        "patch_similarity_mean": (
            round(mean(c.patch_similarity or 0.0 for c in solved), 4) if solved else None
        ),
        "tokens": tokens,
        # Providers split tokens differently (Codex reports no cache-creation tokens and
        # books most context as input), so compare the totals, not the columns.
        "tokens_per_solved": round(tokens["total"] / len(solved)) if solved else None,
        "output_tokens_per_solved": round(tokens["output"] / len(solved)) if solved else None,
        "tokens_median": percentile(per_case_tokens, 0.5),
        "tokens_p90": percentile(per_case_tokens, 0.9),
        "tokens_p95": percentile(per_case_tokens, 0.95),
        "output_tokens_median": percentile(per_case_output, 0.5),
        "cap_hits": sum(1 for c in attempted if c.cap_hit),
        "touched_test_files": sum(1 for c in attempted if c.touched_test_files),
        "host_exec_attempts": sum(1 for c in attempted if c.host_exec_items),
    }


def _kind_line(name: str, st: dict[str, Any]) -> str:
    return f"| {name} | {st['resolved']}/{st['n']} | {_pct(st['rate'])}{_ci(st['ci95'])} |"


def run_markdown(s: RunSummary) -> str:
    m = run_metrics(s)
    o = m["overall"]
    lines = [
        f"# Run {s.run_id}",
        "",
        f"Model `{s.model}` · judge `{s.judge or 'off'}` · harness `{s.harness_git_sha}` · scored {s.scored_at}",
        "",
        "## Resolve rate",
        "",
        "| Slice | Resolved | Rate [95% CI] |",
        "|---|---|---|",
        _kind_line("overall", o),
        *(_kind_line(k, v) for k, v in m["by_kind"].items()),
        *(_kind_line(f"split: {k}", v) for k, v in m["by_split"].items() if v["n"]),
        *(_kind_line(f"author: {k}", v) for k, v in m["by_author_kind"].items()),
        *(_kind_line(f"spec: {k}", v) for k, v in m["by_spec_level"].items()),
        "",
        f"Invalid cases: {o['invalid']} · errors: {o['errors']} · cap hits: {m['cap_hits']} · "
        f"touched test files: {m['touched_test_files']} · host-exec attempts: {m['host_exec_attempts']}",
        "",
        "## Cost and time",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Cost per solved case (headline) | {_money(m['cost_per_solved_case'])} |",
        f"| Cost per solved case, median / p90 | {_money(m['cost_solved_median'])} / {_money(m['cost_solved_p90'])} |",
        f"| Cost per attempted case, median / p90 / p95 | {_money(m['cost_attempted_median'])} / {_money(m['cost_attempted_p90'])} / {_money(m['cost_attempted_p95'])} |",
        f"| Total cost | {_money(m['total_cost_usd'])} |",
        f"| Agent wall-clock per case, median / p90 / p95 | {_num(m['wall_median_s'], ' s')} / {_num(m['wall_p90_s'], ' s')} / {_num(m['wall_p95_s'], ' s')} |",
        f"| Tool calls per case, median / p90 | {_num(m['tool_calls_median'])} / {_num(m['tool_calls_p90'])} |",
        "",
        "## Tokens",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Tokens per solved case | {_num(m['tokens_per_solved'])} |",
        f"| Output tokens per solved case | {_num(m['output_tokens_per_solved'])} |",
        f"| Tokens per case, median / p90 / p95 | {_num(m['tokens_median'])} / {_num(m['tokens_p90'])} / {_num(m['tokens_p95'])} |",
        f"| Total tokens | {_num(m['tokens']['total'])} |",
        f"| Input / output | {_num(m['tokens']['input'])} / {_num(m['tokens']['output'])} |",
        f"| Cache read / write | {_num(m['tokens']['cache_read'])} / {_num(m['tokens']['cache_write'])} |",
        "",
        "Providers split these differently: Codex reports no cache-creation tokens and books",
        "most context as input, so compare totals rather than individual columns.",
        "",
        "## Resolve rate by difficulty",
        "",
        "| Difficulty | Resolved | Rate [95% CI] |",
        "|---|---|---|",
        *[
            _kind_line(k, m["by_difficulty"][k])
            for k in ("easy", "medium", "hard")
            if m["by_difficulty"][k]["n"]
        ],
        "",
        "Rated from each case's own attributes (size, how much the ticket gives away, how many",
        "suites it spans), never from how many models solved it.",
        "",
        "## Quality of solved cases",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| No regression | {_pct(m['no_regression_rate'])} |",
        f"| Lint clean | {_pct(m['lint_clean_rate'])} |",
        f"| Code quality (judge mean) | {_pct(m['code_quality_mean'])} |",
        f"| Patch similarity to human (informational) | {_pct(m['patch_similarity_mean'])} |",
        "",
        "## Per case",
        "",
        "| Case | Kind | Split | Result | Tests | Regr. | Lint | Quality | Sim. | Cost | Agent s | Tools |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(s.cases, key=lambda c: c.case_id):
        result = (
            "✓"
            if c.resolved
            else ("invalid" if c.status == "invalid" else ("error" if c.status == "error" else "✗"))
        )
        if c.cap_hit:
            result += f" ({c.cap_hit})"
        lines.append(
            f"| {c.case_id} | {c.kind} | {c.split} | {result} | {_pct(c.tests_pass)} | "
            f"{_pct(c.no_regression)} | {_pct(c.lint_clean)} | {_pct(c.code_quality)} | "
            f"{_pct(c.patch_similarity)} | {_money(c.cost_usd)} | "
            f"{_num(c.agent_wall_clock_seconds)} | {c.tool_calls} |"
        )
    n = o["n"]
    lines += [
        "",
        "## How to read this",
        "",
        f"With {n} attempted cases the 95% interval on the resolve rate is wide; treat differences "
        "of a few points between runs as noise unless the comparison's paired interval excludes zero. "
        "Cost is the provider's estimate (or the price table) and not what a subscription is billed; "
        "the number that matters for a subscription is quota. Resolved means the case's own tests pass "
        "after the attempt with the test files restored; it does not mean the change is what a reviewer "
        "would merge, which is what the judge column and the regression and lint columns are for.",
    ]
    return "\n".join(lines) + "\n"


def compare_markdown(summaries: list[RunSummary]) -> tuple[str, dict[str, Any]]:
    if len(summaries) < 2:
        raise ValueError("compare needs at least two runs")
    base, *others = summaries
    # Pair only cases every run completed; invalid or errored attempts are not "unsolved".
    case_ids = sorted(
        set.intersection(
            *({c.case_id for c in s.cases if c.status == "completed"} for s in summaries)
        )
    )
    by_run = {s.run_id: {c.case_id: c for c in s.cases} for s in summaries}
    data: dict[str, Any] = {
        "baseline": base.run_id,
        "runs": [s.run_id for s in summaries],
        "common_cases": len(case_ids),
        "regressions": {},
        "improvements": {},
        "diffs": {},
    }
    lines = [
        f"# Comparison: {' vs '.join(s.run_id for s in summaries)}",
        "",
        f"Baseline `{base.run_id}` (`{base.model}`); {len(case_ids)} cases attempted by every run.",
        "",
    ]
    for s in others:
        lost = [
            cid
            for cid in case_ids
            if by_run[base.run_id][cid].resolved and not by_run[s.run_id][cid].resolved
        ]
        won = [
            cid
            for cid in case_ids
            if not by_run[base.run_id][cid].resolved and by_run[s.run_id][cid].resolved
        ]
        data["regressions"][s.run_id] = lost
        data["improvements"][s.run_id] = won
        a = [1.0 if by_run[base.run_id][cid].resolved else 0.0 for cid in case_ids]
        b = [1.0 if by_run[s.run_id][cid].resolved else 0.0 for cid in case_ids]
        diff, ci = paired_bootstrap_diff(a, b)
        data["diffs"][s.run_id] = {"mean_diff": diff, "ci95": ci}
        lines += [f"## Regressions: cases `{base.run_id}` solved that `{s.run_id}` did not", ""]
        lines += [f"- {cid} ({by_run[base.run_id][cid].kind})" for cid in lost] or ["- none"]
        lines += ["", f"Newly solved by `{s.run_id}`: " + (", ".join(won) if won else "none"), ""]
        verdict = "inside the noise" if ci is None or ci[0] <= 0 <= ci[1] else "outside the noise"
        lines += [
            f"Paired resolve-rate difference `{s.run_id}` - `{base.run_id}`: {100 * diff:+.0f} points{_ci(ci)} — {verdict}.",
            "",
        ]
    lines += [
        "## Head to head",
        "",
        "| Run | Model | Resolved | Rate [95% CI] | bug_fix | feature | Cost / solved | Median agent s | Judge quality |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        m = run_metrics(s)
        o = m["overall"]
        lines.append(
            f"| {s.run_id} | {s.model} | {o['resolved']}/{o['n']} | {_pct(o['rate'])}{_ci(o['ci95'])} | {_pct(m['by_kind']['bug_fix']['rate'])} | {_pct(m['by_kind']['feature']['rate'])} | {_money(m['cost_per_solved_case'])} | {_num(m['wall_median_s'])} | {_pct(m['code_quality_mean'])} |"
        )
        data.setdefault("metrics", {})[s.run_id] = m
    lines += [
        "",
        "## Per-case grid",
        "",
        "| Case | Kind | " + " | ".join(s.run_id for s in summaries) + " |",
        "|---|---|" + "---|" * len(summaries),
    ]
    for cid in case_ids:
        cells = []
        for s in summaries:
            c = by_run[s.run_id][cid]
            cells.append(
                (
                    "✓"
                    if c.resolved
                    else (
                        "inv" if c.status == "invalid" else ("err" if c.status == "error" else "✗")
                    )
                )
                + (f" ({c.cap_hit})" if c.cap_hit else "")
            )
        lines.append(f"| {cid} | {by_run[base.run_id][cid].kind} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"With {len(case_ids)} paired cases a difference of a few points is expected from noise alone; only a paired interval that excludes zero is a result.",
        "",
    ]
    return "\n".join(lines), data


def write_run_report(s: RunSummary, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or (results_root() / s.run_id)
    md = out_dir / "summary.md"
    md.write_text(run_markdown(s))
    s.metrics = run_metrics(s)
    js = out_dir / "summary.json"
    js.write_text(json.dumps(s.model_dump(), indent=2) + "\n")
    return js, md


def write_comparison(
    summaries: list[RunSummary],
    out_dir: Path | None = None,
    name_fn: Callable[[list[RunSummary]], str] | None = None,
) -> tuple[Path, Path]:
    out_dir = out_dir or results_root()
    name = name_fn(summaries) if name_fn else "compare-" + "-vs-".join(s.run_id for s in summaries)
    text, data = compare_markdown(summaries)
    md = out_dir / f"{name}.md"
    js = out_dir / f"{name}.json"
    md.write_text(text)
    js.write_text(json.dumps(data, indent=2) + "\n")
    return js, md
