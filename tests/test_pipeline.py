from pathlib import Path
from typing import Any

from eval_harness.collect.cases import Case
from eval_harness.collect.curation import Verdict
from eval_harness.collect.github import PRFile, PullRequest
from eval_harness.collect.linear import Issue
from eval_harness.collect.pipeline import collect
from eval_harness.config import load_repos

CODE = PRFile(path="web/services/x.ts", additions=30, deletions=5)
TEST = PRFile(path="web/services/__tests__/x.test.ts", additions=40, deletions=0)


def _pr(n: int, key: str, merged: str = "2026-09-02T00:00:00Z") -> PullRequest:
    return PullRequest(
        number=n,
        title=f"{key}: t",
        body="",
        head_ref=f"tony/{key}-x",
        author="tony",
        merged_at=merged,
        created_at="2026-09-01T00:00:00Z",
        merge_commit="c" * 40,
        files=[CODE, TEST],
        review_count=1,
        comment_count=0,
    )


def _issue(key: str, labels: list[str] | None = None) -> Issue:
    return Issue(
        key=key,
        number=int(key.split("-")[1]),
        title="t",
        description="d" * 300,
        labels=labels or [],
        created_at="2026-08-30T00:00:00Z",
        started_at=None,
        completed_at=None,
        comments=[],
        children=[],
    )


def _fake_builder(repo: Any, pr: PullRequest, issue: Issue, **kw: Any) -> Case:
    return Case(
        case_id=issue.key,
        repo=repo.github,
        kind=kw["kind"],
        base_commit="a" * 40,
        task_prompt="t",
        test_command="x",
        test_files_added=[],
        test_files=[TEST.path],
        code_files=[CODE.path],
        human_patch="",
        human_test_patch="",
        files_changed=2,
        lines_changed=75,
        human_cycle_time_hours=1.0,
        review_comment_count=1,
        merged_at=pr.merged_at,
        pr_number=pr.number,
        merge_commit=pr.merge_commit,
        test_runner="unit",
        files_dropped=[],
        tier=kw["tier"],
        kind_source=kw["kind_source"],
        author=pr.author,
        author_kind=kw["author_kind"],
        spec_level_heuristic=kw["spec_level_heuristic"],
        ticket_created_at=issue.created_at,
        ticket_started_at=None,
        human_cycle_time_source="created_at",
    )


class _FakeTickets:
    """The protocol the pipeline now talks to, with the issues the test already built."""

    def __init__(self, issues: dict[str, object], team: str) -> None:
        self.issues, self.team = issues, team

    def key_for(self, pr: object) -> tuple[str | None, str | None]:
        from eval_harness.collect.join import ticket_for

        return ticket_for(pr, self.team)  # type: ignore[arg-type]

    def fetch(self, keys: list[str], **kw: object) -> dict[str, object]:
        return self.issues


def test_collect_applies_rules_curation_and_kind_resolution(tmp_path: Path) -> None:
    repo = load_repos()["demo-app"]
    prs = [
        _pr(1, "DEMO-1", "2026-05-01T00:00:00Z"),  # tier A, label Bug
        _pr(2, "DEMO-2", "2026-06-01T00:00:00Z"),  # tier A, reviewer kind feature
        _pr(3, "DEMO-3", "2026-07-01T00:00:00Z"),  # tier A, needs classifier
        _pr(4, "DEMO-4"),  # tier X
        _pr(5, "DEMO-5"),  # uncurated
        _pr(6, "DEMO-6"),
        _pr(7, "DEMO-6"),  # multi-PR ticket -> S6
    ]
    issues = {
        "DEMO-1": _issue("DEMO-1", ["Bug"]),
        "DEMO-2": _issue("DEMO-2"),
        "DEMO-3": _issue("DEMO-3"),
        "DEMO-4": _issue("DEMO-4"),
        "DEMO-5": _issue("DEMO-5"),
        "DEMO-6": _issue("DEMO-6"),
    }
    verdicts = {
        "DEMO-1": Verdict(tier="A", reason="", pr=1, kind_review="bug_fix"),
        "DEMO-2": Verdict(tier="A", reason="", pr=2, kind_review="feature"),
        "DEMO-3": Verdict(tier="A", reason="", pr=3, kind_review="-"),
        "DEMO-4": Verdict(tier="X", reason="trivial", pr=4),
        "DEMO-6": Verdict(tier="A", reason="", pr=6, kind_review="bug_fix"),
    }
    classified: list[str] = []

    def classify(title: str, description: str) -> str:
        classified.append(title)
        return "bug_fix"

    report = collect(
        repo,
        since="2026-03-09",
        until="2026-09-09",
        verdicts=verdicts,
        fetch_prs=lambda *a, **k: prs,
        tickets=_FakeTickets(issues, repo.linear_team),
        classify=classify,
        case_builder=_fake_builder,
        cases_dir=tmp_path / "cases",
        splits_path=tmp_path / "splits.json",
    )
    assert report.match.total == 7 and report.match.with_key == 7
    assert report.rejections == {"S6": 2}
    assert report.accepted == 5
    assert report.written == ["DEMO-1", "DEMO-2", "DEMO-3"]
    assert report.curated_out == {"X": 1} and report.uncurated == ["DEMO-5"]
    assert report.kind_split == {"bug_fix": 2, "feature": 1}
    assert report.kind_sources == {"label": 1, "review": 1, "llm": 1} and classified == ["t"]
    assert (tmp_path / "cases" / "DEMO-3.json").exists() and (tmp_path / "splits.json").exists()
    assert report.splits == {"dev": 2, "holdout": 1}
    assert any("below the 70%" not in w for w in report.warnings)
    assert any("bug_fix" in w for w in report.warnings)  # fewer than 20 per kind
