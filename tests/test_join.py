from eval_harness.collect.github import PullRequest
from eval_harness.collect.join import is_bot_pr, match_stats, ticket_for


def _pr(n: int, branch: str, title: str = "", body: str = "", author: str = "dev") -> PullRequest:
    return PullRequest(
        number=n,
        title=title,
        body=body,
        head_ref=branch,
        author=author,
        merged_at="2026-09-02T00:00:00Z",
        created_at="2026-09-01T00:00:00Z",
        merge_commit="c" * 40,
        files=[],
        review_count=0,
        comment_count=0,
    )


def test_ticket_for_prefers_branch_then_title_then_body() -> None:
    assert ticket_for(_pr(1, "tony/DEMO-12-x", "DEMO-99: t", "DEMO-7"), "DEMO") == (
        "DEMO-12",
        "branch",
    )
    assert ticket_for(_pr(2, "feat/x", "demo-99: t", "DEMO-7"), "DEMO") == ("DEMO-99", "title")
    assert ticket_for(_pr(3, "feat/x", "t", "Closes DEMO-7"), "DEMO") == ("DEMO-7", "body")
    assert ticket_for(_pr(4, "feat/x", "t", "none"), "DEMO") == (None, None)


def test_bot_detection_and_match_stats() -> None:
    prs = [
        _pr(1, "tony/DEMO-1-a"),
        _pr(2, "test/auto-daily-coverage"),
        _pr(3, "neo/pr3059-qa-fixback", body="DEMO-3"),
        _pr(4, "feat/no-ticket-cleanup"),
        _pr(5, "x/y", author="github-actions[bot]"),
    ]
    assert [is_bot_pr(p) for p in prs] == [False, True, True, True, True]
    stats = match_stats(prs, "DEMO")
    assert (stats.total, stats.with_key, stats.human_total, stats.human_with_key) == (5, 2, 1, 1)
    assert stats.rate == 0.4 and stats.human_rate == 1.0
    assert stats.by_source == {"branch": 1, "body": 1}
