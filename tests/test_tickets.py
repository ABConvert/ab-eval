"""Ticket sources: how a PR names its issue, and what a case gets back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_harness.collect.github import PullRequest
from eval_harness.collect.tickets import GitHubIssues, LinearTickets, source_for


def pr(**kw: object) -> PullRequest:
    base: dict[str, object] = {
        "number": 1,
        "title": "",
        "body": "",
        "head_ref": "",
        "author": "a",
        "merged_at": "",
        "created_at": "",
        "merge_commit": "x",
        "files": [],
        "review_count": 0,
        "comment_count": 0,
    }
    base.update(kw)
    return PullRequest.model_validate(base)


class _Repo:
    def __init__(self, team: str = "") -> None:
        self.github, self.linear_team = "acme/widget", team


def test_closing_keyword_in_the_body_names_the_issue() -> None:
    gh = GitHubIssues("acme/widget")
    for body in ("Fixes #1284", "closes  #1284", "Resolves: #1284", "fixed #1284"):
        assert gh.key_for(pr(body=body)) == ("#1284", "body"), body


def test_a_bare_number_in_the_title_is_accepted() -> None:
    assert GitHubIssues("acme/widget").key_for(pr(title="Tidy the parser (#99)")) == (
        "#99",
        "title",
    )


def test_a_bare_number_in_the_body_is_not_enough() -> None:
    """PR bodies mention unrelated issues constantly; a wrong ticket is worse than no case."""
    assert GitHubIssues("acme/widget").key_for(pr(body="see also #55")) == (None, None)


def test_no_reference_at_all() -> None:
    assert GitHubIssues("acme/widget").key_for(pr(title="tidy up", body="no refs")) == (None, None)


def test_linear_reads_the_branch_name() -> None:
    assert LinearTickets("DEMO").key_for(pr(head_ref="feat/DEMO-42-thing")) == ("DEMO-42", "branch")


def test_source_is_chosen_by_whether_a_team_key_is_declared() -> None:
    assert isinstance(source_for(_Repo("DEMO")), LinearTickets)
    assert isinstance(source_for(_Repo()), GitHubIssues)


def test_fetch_maps_the_api_shape_onto_a_ticket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "title": "Parser drops trailing commas",
        "body": "Steps to reproduce ...",
        "labels": [{"name": "bug"}],
        "created_at": "2026-01-02T03:04:05Z",
        "closed_at": "2026-01-03T00:00:00Z",
    }

    def fake(argv: list[str], **kw: object) -> object:
        class R:
            returncode = 0
            stdout = json.dumps(
                [{"body": "me too"}] if any("comments" in a for a in argv) else payload
            )
            stderr = ""

        return R()

    monkeypatch.setattr("subprocess.run", fake)
    got = GitHubIssues("acme/widget").fetch(["#7"], cache_dir=tmp_path)
    issue = got["#7"]
    assert (issue.key, issue.number, issue.title) == ("#7", 7, "Parser drops trailing commas")
    assert issue.labels == ["bug"] and issue.comments == ["me too"]
    assert issue.started_at is None  # GitHub has no equivalent; PR dates carry cycle time
    assert issue.children == []  # no sub-issues, so the epic rule never fires


def test_a_pull_request_is_not_a_ticket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The issues endpoint returns PRs too; treating one as a ticket would be nonsense."""

    def fake(argv: list[str], **kw: object) -> object:
        class R:
            returncode = 0
            stdout = json.dumps({"title": "a PR", "pull_request": {"url": "..."}})
            stderr = ""

        return R()

    monkeypatch.setattr("subprocess.run", fake)
    assert GitHubIssues("acme/widget").fetch(["#7"], cache_dir=tmp_path) == {}


def test_fetch_uses_the_cache_on_a_second_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"n": 0}

    def fake(argv: list[str], **kw: object) -> object:
        calls["n"] += 1

        class R:
            returncode = 0
            stdout = (
                "[]"
                if any("comments" in a for a in argv)
                else json.dumps({"title": "t", "body": "b", "labels": [], "created_at": "x"})
            )
            stderr = ""

        return R()

    monkeypatch.setattr("subprocess.run", fake)
    src = GitHubIssues("acme/widget")
    src.fetch(["#7"], cache_dir=tmp_path)
    first = calls["n"]
    src.fetch(["#7"], cache_dir=tmp_path)
    assert calls["n"] == first, "second fetch should have come from the cache"


def test_collecting_from_linear_without_the_key_names_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The unguarded `os.environ[...]` was a KeyError with no noun in it."""
    from eval_harness.collect import linear

    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.setattr(
        linear.httpx, "post", lambda *a, **k: pytest.fail("Linear was called without a key")
    )
    with pytest.raises(RuntimeError) as e:
        LinearTickets("DEMO").fetch(["DEMO-42"], cache_dir=tmp_path)
    assert "LINEAR_API_KEY" in str(e.value)
    assert "linear_team" in str(e.value), "and why this repository needs it"
    assert "Setup" in str(e.value), "where being set is reported"


def test_building_one_case_from_a_pr_fails_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single-PR path reaches Linear through its own request, and used to crash its own way."""
    from eval_harness.collect import single

    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.setattr(
        single.httpx, "post", lambda *a, **k: pytest.fail("Linear was called without a key")
    )
    with pytest.raises(RuntimeError) as e:
        single._linear_issue("DEMO-42")
    assert "LINEAR_API_KEY" in str(e.value)


def test_the_key_is_passed_on_when_it_is_there_and_never_said_out_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guarding the read must not change what a configured machine does with it."""
    from eval_harness.collect import linear

    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_sentinel")
    sent: dict[str, object] = {}

    class R:
        @staticmethod
        def raise_for_status() -> None: ...

        @staticmethod
        def json() -> dict[str, object]:
            return {"data": {"issues": {"nodes": []}}}

    def fake(url: str, **kw: object) -> R:
        sent.update(kw)
        return R()

    monkeypatch.setattr(linear.httpx, "post", fake)
    linear._post("query", {})
    assert sent["headers"] == {
        "Authorization": "lin_api_sentinel",
        "Content-Type": "application/json",
    }
