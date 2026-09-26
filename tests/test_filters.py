from eval_harness.collect.filters import decide, reverted_numbers
from eval_harness.collect.github import PRFile, PullRequest
from eval_harness.collect.linear import Issue
from eval_harness.config import load_repos

REPO = load_repos()["demo-app"]
CODE = PRFile(path="web/services/x.ts", additions=30, deletions=5)
TEST = PRFile(path="web/services/__tests__/x.test.ts", additions=40, deletions=0)


def _pr(
    n: int = 1,
    branch: str = "tony/DEMO-10-x",
    title: str = "DEMO-10: fix",
    files: list[PRFile] | None = None,
    created: str = "2026-09-01T00:00:00Z",
    merged: str = "2026-09-02T00:00:00Z",
) -> PullRequest:
    return PullRequest(
        number=n,
        title=title,
        body="",
        head_ref=branch,
        author="tony",
        merged_at=merged,
        created_at=created,
        merge_commit="c" * 40,
        files=[CODE, TEST] if files is None else files,
        review_count=0,
        comment_count=0,
    )


def _issue(
    desc_len: int = 300, children: list[str] | None = None, created: str = "2026-08-30T00:00:00Z"
) -> Issue:
    return Issue(
        key="DEMO-10",
        number=10,
        title="t",
        description="x" * desc_len,
        labels=[],
        created_at=created,
        started_at=None,
        completed_at=None,
        comments=[],
        children=children or [],
    )


def test_clean_pr_is_accepted() -> None:
    d = decide(_pr(), _issue(), repo=REPO, prs_for_key=1, reverted=set())
    assert d.accept and d.key == "DEMO-10" and d.detail == "via branch"


def test_rules_fire_in_order() -> None:
    assert decide(_pr(), _issue(), repo=REPO, prs_for_key=1, reverted={1}).rule == "S1"
    assert (
        decide(
            _pr(branch="feat/x", title="fix"), None, repo=REPO, prs_for_key=0, reverted=set()
        ).rule
        == "S2"
    )
    assert decide(_pr(), _issue(desc_len=50), repo=REPO, prs_for_key=1, reverted=set()).rule == "S2"
    assert decide(_pr(), _issue(), repo=REPO, prs_for_key=2, reverted=set()).rule == "S6"
    assert (
        decide(_pr(), _issue(children=["DEMO-11"]), repo=REPO, prs_for_key=1, reverted=set()).rule
        == "R1"
    )
    assert (
        decide(
            _pr(title="DEMO-10 + DEMO-12: two"), _issue(), repo=REPO, prs_for_key=1, reverted=set()
        ).rule
        == "R2"
    )
    assert (
        decide(
            _pr(branch="tony/DEMO-10-x", title="DEMO-12: sub"),
            _issue(),
            repo=REPO,
            prs_for_key=1,
            reverted=set(),
        ).rule
        == "R3"
    )
    assert (
        decide(
            _pr(), _issue(created="2026-09-05T00:00:00Z"), repo=REPO, prs_for_key=1, reverted=set()
        ).rule
        == "R4"
    )
    assert (
        decide(_pr(files=[CODE]), _issue(), repo=REPO, prs_for_key=1, reverted=set()).rule == "S3"
    )
    big = [PRFile(path=f"web/services/f{i}.ts", additions=50, deletions=0) for i in range(10)] + [
        TEST
    ]
    assert decide(_pr(files=big), _issue(), repo=REPO, prs_for_key=1, reverted=set()).rule == "S4"
    tiny_test = [CODE, PRFile(path="web/services/__tests__/x.test.ts", additions=3, deletions=0)]
    assert (
        decide(_pr(files=tiny_test), _issue(), repo=REPO, prs_for_key=1, reverted=set()).rule
        == "R7"
    )
    tiny_code = [PRFile(path="web/services/x.ts", additions=2, deletions=1), TEST]
    assert (
        decide(_pr(files=tiny_code), _issue(), repo=REPO, prs_for_key=1, reverted=set()).rule
        == "R8"
    )


def test_reverted_numbers_within_window() -> None:
    a = _pr(n=100, merged="2026-06-01T00:00:00Z")
    r = _pr(n=101, branch="revert-100", title="Revert #100", merged="2026-06-10T00:00:00Z")
    late = _pr(
        n=102, branch="revert-100-late", title="Revert #100 again", merged="2026-08-10T00:00:00Z"
    )
    assert reverted_numbers([a, r]) == {100}
    assert reverted_numbers([a, late]) == set()
