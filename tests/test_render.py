from pathlib import Path

from eval_harness.report.render import compare_markdown, run_markdown, run_metrics, write_comparison
from eval_harness.report.summary import CaseScore, RunSummary


def _case(
    cid: str, resolved: bool, kind: str = "bug_fix", cost: float = 1.0, status: str = "completed"
) -> CaseScore:
    return CaseScore(
        case_id=cid,
        kind=kind,
        tier="A",
        author_kind="human",
        spec_level="symptom",
        split="dev",
        status=status,
        resolved=resolved,
        tests_pass=1.0 if resolved else 0.5,
        no_regression=1.0 if resolved else None,
        lint_clean=1.0 if resolved else None,
        patch_similarity=0.5,
        code_quality=0.8 if resolved else None,
        cost_usd=cost,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        wall_clock_seconds=100.0,
        agent_wall_clock_seconds=50.0,
        turns=3,
        tool_calls=4,
        cap_hit=None,
        touched_test_files=False,
        host_exec_items=0,
        files_touched=1,
        diff_lines=10,
    )


def _run(rid: str, results: dict[str, bool]) -> RunSummary:
    return RunSummary(
        run_id=rid,
        model=f"model-{rid}",
        model_config_snapshot={},
        harness_git_sha="abc",
        judge="j",
        judge_config=None,
        scored_at="2026-09-10T00:00:00+00:00",
        cases=[
            _case(cid, ok, kind="feature" if i % 2 else "bug_fix")
            for i, (cid, ok) in enumerate(results.items())
        ],
    )


def test_run_metrics_and_markdown() -> None:
    s = _run("a", {"C1": True, "C2": False, "C3": True, "C4": False})
    s.cases.append(_case("C5", False, status="invalid"))
    m = run_metrics(s)
    assert m["overall"]["n"] == 4 and m["overall"]["resolved"] == 2 and m["overall"]["invalid"] == 1
    assert m["cost_per_solved_case"] == 2.0 and m["by_kind"]["bug_fix"]["resolved"] == 2
    md = run_markdown(s)
    assert "| overall | 2/4 | 50%" in md and "| C5 | bug_fix | dev | invalid |" in md
    assert "How to read this" in md


def test_compare_lists_regressions_first(tmp_path: Path) -> None:
    base = _run("base", {"C1": True, "C2": True, "C3": False, "C4": False})
    cand = _run("cand", {"C1": True, "C2": False, "C3": True, "C4": True})
    text, data = compare_markdown([base, cand])
    assert data["regressions"]["cand"] == ["C2"] and data["improvements"]["cand"] == ["C3", "C4"]
    assert text.index("Regressions") < text.index("Head to head") < text.index("Per-case grid")
    js, md = write_comparison([base, cand], out_dir=tmp_path)
    assert js.exists() and md.exists() and "compare-base-vs-cand" in md.name
