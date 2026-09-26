from pathlib import Path

from eval_harness.collect.cases import Case
from eval_harness.collect.split import assign_splits, load_splits, write_splits


def _case(cid: str, merged: str) -> Case:
    return Case(
        case_id=cid,
        repo="r",
        kind="bug_fix",
        base_commit="a" * 40,
        task_prompt="t",
        test_command="x",
        test_files_added=[],
        test_files=["web/a.test.ts"],
        code_files=["web/a.ts"],
        human_patch="",
        human_test_patch="",
        files_changed=2,
        lines_changed=3,
        human_cycle_time_hours=1.0,
        review_comment_count=0,
        merged_at=merged,
        pr_number=1,
        merge_commit="b" * 40,
        test_runner="unit",
        files_dropped=[],
        tier="A",
        kind_source="review",
        author="x",
        author_kind="human",
        spec_level_heuristic="symptom",
        ticket_created_at="2026-01-01T00:00:00Z",
        ticket_started_at=None,
        human_cycle_time_source="created_at",
    )


def test_holdout_is_the_most_recent_30_percent(tmp_path: Path) -> None:
    cases = [_case(f"C{i}", f"2026-0{1 + i // 9}-{1 + i % 9:02d}T00:00:00Z") for i in range(10)]
    assignments = assign_splits(cases)
    assert sum(v == "holdout" for v in assignments.values()) == 3
    ordered = sorted(cases, key=lambda c: c.merged_at)
    assert all(assignments[c.case_id] == "dev" for c in ordered[:7])
    assert all(assignments[c.case_id] == "holdout" for c in ordered[7:])
    path = write_splits(assignments, tmp_path / "splits.json")
    loaded = load_splits(path)
    assert len(loaded["dev"]) == 7 and len(loaded["holdout"]) == 3
