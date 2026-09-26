"""Where a task description comes from.

A case needs the ticket that asked for the change: its text becomes the prompt, and its dates
say how long a human took. Which tracker holds it is the team's business, not the harness's, so
both paths meet at one protocol. GitHub Issues is the default because `gh` is already required
and most teams have it; Linear is opt-in through `linear_team` in repos.yaml.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Protocol

from eval_harness.collect.github import PullRequest, load_cache, save_cache
from eval_harness.collect.linear import Issue

# "Fixes #12", "closes  #12", "Resolves: #12" — the forms GitHub itself closes an issue on.
CLOSES = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[:\s]*#(\d+)", re.I)
HASH = re.compile(r"#(\d+)\b")


class TicketSource(Protocol):
    """How PRs name their tickets, and how to read the tickets they name."""

    def key_for(self, pr: PullRequest) -> tuple[str | None, str | None]:
        """(key, where it was found) — branch, title or body — or (None, None)."""

    def fetch(self, keys: list[str], *, cache_dir: Path, refresh: bool = False) -> dict[str, Issue]:
        """Every ticket that resolves, keyed as `key_for` returns them."""


class LinearTickets:
    """Linear, via the GraphQL API. Keys look like ENG-1284 and live in the branch name."""

    def __init__(self, team: str) -> None:
        self.team = team

    def key_for(self, pr: PullRequest) -> tuple[str | None, str | None]:
        from eval_harness.collect.join import ticket_for

        return ticket_for(pr, self.team)

    def fetch(self, keys: list[str], *, cache_dir: Path, refresh: bool = False) -> dict[str, Issue]:
        from eval_harness.collect.linear import fetch_issues

        numbers = sorted({int(k.split("-")[1]) for k in keys if "-" in k})
        return fetch_issues(self.team, numbers, cache_dir=cache_dir, refresh=refresh)


class GitHubIssues:
    """GitHub Issues, via `gh api`. Keys look like #1284.

    A PR names its issue the way GitHub itself does — "Fixes #1284" in the body — falling back
    to a bare `#1284` in the title. A bare number in the *body* is deliberately not enough: PR
    bodies quote unrelated issues all the time, and a wrong ticket is worse than no case.
    """

    def __init__(self, github: str) -> None:
        self.github = github

    def key_for(self, pr: PullRequest) -> tuple[str | None, str | None]:
        m = CLOSES.search(pr.body or "")
        if m:
            return f"#{m.group(1)}", "body"
        m = HASH.search(pr.title or "")
        if m:
            return f"#{m.group(1)}", "title"
        return None, None

    def fetch(self, keys: list[str], *, cache_dir: Path, refresh: bool = False) -> dict[str, Issue]:
        numbers = sorted({int(k.lstrip("#")) for k in keys if k.lstrip("#").isdigit()})
        name = f"gh-issues-{self.github.replace('/', '_')}-{len(numbers)}"
        if not refresh:
            cached = load_cache(cache_dir, name, Issue)
            if cached is not None:
                return {i.key: i for i in cached}
        issues = [i for n in numbers if (i := self._one(n)) is not None]
        save_cache(cache_dir, name, issues)
        return {i.key: i for i in issues}

    def _one(self, number: int) -> Issue | None:
        proc = subprocess.run(
            ["gh", "api", f"repos/{self.github}/issues/{number}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        raw: dict[str, Any] = json.loads(proc.stdout)
        if "pull_request" in raw:
            return None  # the API returns PRs from this endpoint too; they are not tickets
        return Issue(
            key=f"#{number}",
            number=number,
            title=raw.get("title") or "",
            description=raw.get("body") or "",
            labels=[label["name"] for label in raw.get("labels", [])],
            created_at=raw.get("created_at") or "",
            # GitHub has no "started"; the PR's own dates carry cycle time instead.
            started_at=None,
            completed_at=raw.get("closed_at"),
            comments=self._comments(number),
            children=[],  # no sub-issues; the epic rule simply never fires
        )

    def _comments(self, number: int) -> list[str]:
        proc = subprocess.run(
            ["gh", "api", f"repos/{self.github}/issues/{number}/comments", "--paginate"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return []
        try:
            return [c.get("body") or "" for c in json.loads(proc.stdout)]
        except json.JSONDecodeError:
            return []


def source_for(repo: Any) -> TicketSource:
    """Linear when the repo declares a team key, GitHub Issues otherwise."""
    team = getattr(repo, "linear_team", "") or ""
    return LinearTickets(team) if team else GitHubIssues(repo.github)
