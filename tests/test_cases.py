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


def _case() -> Case:
    return Case(
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


def test_case_roundtrip(tmp_path: Path) -> None:
    case = _case()
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


DIFF_WITH_DATA = """\
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-x = 1
+x = 2
diff --git a/tests/test_app.py b/tests/test_app.py
--- a/tests/test_app.py
+++ b/tests/test_app.py
@@ -1 +1 @@
-assert x == 1
+assert x == 2
diff --git a/schema/user.schema.json b/schema/user.schema.json
--- a/schema/user.schema.json
+++ b/schema/user.schema.json
@@ -1 +1 @@
-{"required": []}
+{"required": ["scope"]}
diff --git a/docs/logo.png b/docs/logo.png
index 1111111..2222222 100644
Binary files a/docs/logo.png and b/docs/logo.png differ
"""


def test_data_files_travel_with_the_reference_but_not_as_code() -> None:
    """A test that read a schema the case builder dropped failed against its own fix."""
    from eval_harness.collect.cases import split_diff, text_chunks

    def is_test(p: str) -> bool:
        return p.startswith("tests/")

    def is_code(p: str) -> bool:
        return p.endswith(".py") and not is_test(p)

    code, _test, code_files, _tests, dropped = split_diff(DIFF_WITH_DATA, is_test, is_code)
    assert code_files == ["src/app.py"]
    assert dropped == ["schema/user.schema.json", "docs/logo.png"]

    extra = text_chunks(DIFF_WITH_DATA, dropped)
    assert '+{"required": ["scope"]}' in extra
    assert "Binary files" not in extra, "git apply cannot take a binary chunk without its payload"
    assert "x = 2" not in extra

    case = _case().model_copy(update={"human_patch": code, "human_extra_patch": extra})
    assert case.reference_patch() == code + extra
    assert _case().reference_patch() == _case().human_patch, "older case files have no extra"
