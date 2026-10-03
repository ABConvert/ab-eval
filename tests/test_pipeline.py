from pathlib import Path
from typing import Any

import pytest

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


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway data root: collect writes its pending-curation list there."""
    from eval_harness import paths

    monkeypatch.setenv("ABEVAL_DATA_ROOT", str(tmp_path / "root"))
    paths.reset_cache()
    return tmp_path / "root"


def test_collect_applies_rules_curation_and_kind_resolution(
    tmp_path: Path, data_root: Path
) -> None:
    repo = load_repos(Path(__file__).parent / "fixtures/config/repos.yaml")["demo-app"]
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


def _run(repo: Any, prs: list[PullRequest], issues: dict[str, Any], tmp: Path, **kw: Any) -> Any:
    return collect(
        repo,
        since="2026-03-09",
        until="2026-09-09",
        fetch_prs=lambda *a, **k: prs,
        tickets=_FakeTickets(issues, repo.linear_team),
        classify=lambda t, d: "bug_fix",
        case_builder=_fake_builder,
        cases_dir=tmp / "cases",
        splits_path=tmp / "splits.json",
        **kw,
    )


def test_a_fresh_data_root_collects_without_a_verdicts_file(
    tmp_path: Path, data_root: Path, demo_repo: Any
) -> None:
    """It used to raise FileNotFoundError: nothing ever creates verdicts.json for you."""
    report = _run(demo_repo, [_pr(5, "DEMO-5")], {"DEMO-5": _issue("DEMO-5")}, tmp_path)
    assert report.uncurated == ["DEMO-5"]
    assert [c.pr for c in report.pending] == [5]
    assert any("eval-harness curate" in w for w in report.warnings)


def test_curating_the_pending_list_makes_the_next_collect_write_the_case(
    tmp_path: Path, data_root: Path, demo_repo: Any
) -> None:
    from eval_harness.collect.curation import load_pending, load_verdicts, record_verdicts

    prs, issues = [_pr(5, "DEMO-5")], {"DEMO-5": _issue("DEMO-5")}
    _run(demo_repo, prs, issues, tmp_path, dry_run=True)
    assert [c.key for c in load_pending("demo-app")] == ["DEMO-5"]

    unknown = record_verdicts("demo-app", ["demo-5", "DEMO-99"], tier="A", reason="ok")
    assert unknown == ["DEMO-99"]
    assert load_verdicts()["DEMO-5"].pr == 5

    report = _run(demo_repo, prs, issues, tmp_path)
    assert report.written == ["DEMO-5"] and report.pending == []


def test_selection_limits_come_from_repos_yaml(
    tmp_path: Path, data_root: Path, demo_repo: Any
) -> None:
    """S6 was a constant: a team that ships one ticket as two PRs could not collect at all."""
    from eval_harness.collect.curation import Verdict

    prs = [_pr(6, "DEMO-6"), _pr(7, "DEMO-6")]
    issues = {"DEMO-6": _issue("DEMO-6")}
    verdicts = {"DEMO-6": Verdict(tier="A", reason="", pr=6, kind_review="bug_fix")}
    assert _run(demo_repo, prs, issues, tmp_path, verdicts=verdicts).rejections == {"S6": 2}

    loose = demo_repo.model_copy(
        update={"selection": demo_repo.selection.model_copy(update={"max_prs_per_ticket": 2})}
    )
    assert _run(loose, prs, issues, tmp_path, verdicts=verdicts).rejections == {}


def test_the_report_names_what_each_rule_means_and_what_awaits_curation(
    tmp_path: Path, data_root: Path, demo_repo: Any
) -> None:
    from eval_harness.collect.report import render_text

    prs = [_pr(5, "DEMO-5"), _pr(6, "DEMO-6"), _pr(7, "DEMO-6")]
    issues = {"DEMO-5": _issue("DEMO-5"), "DEMO-6": _issue("DEMO-6")}
    text = render_text(_run(demo_repo, prs, issues, tmp_path))
    assert "S6" in text and "max_prs_per_ticket" in text
    assert "awaiting curation" in text and "#5" in text


@pytest.mark.parametrize("dry_run", [False, True])
def test_secrets_are_rejected_before_classification(
    tmp_path: Path,
    data_root: Path,
    demo_repo: Any,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
) -> None:
    from eval_harness.collect import classify as classifier
    from eval_harness.collect.single import build_task_prompt
    from eval_harness.config import load_models

    issue = _issue("DEMO-1")
    issue.description += " credential: " + "ghp_" + "A" * 36
    sent: list[str] = []

    async def provider(prompt: str, **kw: Any) -> str:
        sent.append(prompt)
        return "bug_fix"

    def builder(repo: Any, pr: PullRequest, issue: Issue, **kw: Any) -> Case:
        build_task_prompt(issue)
        return _fake_builder(repo, pr, issue, **kw)

    monkeypatch.setattr(classifier, "oneshot", provider)
    report = collect(
        demo_repo,
        since="2026-03-09",
        until="2026-09-09",
        dry_run=dry_run,
        verdicts={"DEMO-1": Verdict(tier="A", reason="", pr=1, kind_review="-")},
        fetch_prs=lambda *a, **k: [_pr(1, "DEMO-1")],
        tickets=_FakeTickets({"DEMO-1": issue}, demo_repo.linear_team),
        classifier_model=load_models(Path(__file__).parent / "fixtures/config/models.yaml")[
            "classifier"
        ],
        case_builder=builder,
        save_pending_list=False,
    )
    assert sent == []
    assert report.secrets_hits == ["DEMO-1: github_token"]
    assert report.written == []


async def test_classifier_redacts_before_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval_harness.collect import classify as classifier
    from eval_harness.config import load_models

    sent: list[str] = []

    async def provider(prompt: str, **kw: Any) -> str:
        sent.append(prompt)
        return "bug_fix"

    monkeypatch.setattr(classifier, "oneshot", provider)
    model = load_models(Path(__file__).parent / "fixtures/config/models.yaml")["classifier"]
    await classifier.classify_kind_async(
        "Contact bob@example.com",
        "See https://linear.app/acme/issue/DEMO-1",
        model=model,
    )
    assert len(sent) == 1
    assert "bob@example.com" not in sent[0]
    assert "linear.app" not in sent[0]
