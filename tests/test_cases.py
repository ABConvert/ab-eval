from pathlib import Path

from eval_harness.collect.cases import Case, count_changed_lines, load_case, save_case, split_diff
from eval_harness.config import load_repos

# Synthetic: a real diff from a private repository is that repository's source code,
# which a public test suite should not carry.
FIX = Path(__file__).parent / "fixtures" / "code-and-test.diff"


def test_split_diff_separates_code_and_tests() -> None:
    cfg = load_repos()["demo-app"]
    code, test, code_files, test_files, dropped = split_diff(
        FIX.read_text(), cfg.is_test_file, cfg.is_code_file
    )
    assert code_files == ["web/services/demo/ExampleService.ts"]
    assert test_files == ["web/services/demo/__tests__/ExampleService.test.ts"]
    assert dropped == []
    assert code.startswith("diff --git a/web/services/demo/ExampleService.ts")
    assert "ExampleService.test.ts" not in code
    assert test.startswith("diff --git a/web/services/demo/__tests__/ExampleService.test.ts")
    assert "FIRST_RETURNING_ORDER_INDEX" not in test


def test_count_changed_lines_matches_git_stat() -> None:
    assert count_changed_lines(FIX.read_text()) == 47 + 8  # code + test lines


def test_case_roundtrip(tmp_path: Path) -> None:
    case = Case(
        case_id="DEMO-0",
        repo="acme/demo-app",
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
        merged_at="2026-09-02T13:28:34Z",
        pr_number=1,
        merge_commit="b" * 40,
        test_runner="unit",
        files_dropped=[],
        tier="A",
        kind_source="review",
        author="x",
        author_kind="human",
        spec_level_heuristic="symptom",
        ticket_created_at="2026-09-02T06:00:00Z",
        ticket_started_at=None,
        human_cycle_time_source="created_at",
    )
    path = save_case(case, tmp_path)
    assert path.name == "DEMO-0.json"
    assert load_case("DEMO-0", tmp_path) == case


def test_agent_authors_come_from_the_repo_not_the_harness() -> None:
    """Whose accounts are bots is a property of a repository, not of this tool."""
    from eval_harness.collect.single import AGENT_AUTHORS
    from eval_harness.config import load_repos
    from tests.conftest import FIXTURES

    assert AGENT_AUTHORS == {}, "no company's roster belongs in the source"
    repo = load_repos(FIXTURES / "config" / "repos.yaml")["demo-app"]
    assert repo.agent_authors == {"demo-bot": "agent", "demo-unclear": "unclear"}
