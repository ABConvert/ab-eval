from __future__ import annotations

import re

from pydantic import BaseModel

from eval_harness.collect.github import PullRequest

BOT_BRANCH = re.compile(r"^(test/auto|neo/|codex/|agent/|dependabot|renovate)|no-ticket", re.I)


def is_bot_pr(pr: PullRequest) -> bool:
    return bool(BOT_BRANCH.search(pr.head_ref)) or "[bot]" in pr.author


def ticket_for(pr: PullRequest, team: str) -> tuple[str | None, str | None]:
    """(key, source) with source in branch|title|body, first match wins."""
    pattern = re.compile(rf"\b{re.escape(team)}-(\d+)\b", re.I)
    for source, text in (("branch", pr.head_ref), ("title", pr.title), ("body", pr.body)):
        m = pattern.search(text or "")
        if m:
            return f"{team}-{m.group(1)}", source
    return None, None


class MatchStats(BaseModel):
    total: int
    with_key: int
    rate: float
    human_total: int
    human_with_key: int
    human_rate: float
    by_source: dict[str, int]


def match_stats(prs: list[PullRequest], team: str) -> MatchStats:
    by_source: dict[str, int] = {}
    with_key = human_total = human_with_key = 0
    for pr in prs:
        key, source = ticket_for(pr, team)
        human = not is_bot_pr(pr)
        human_total += human
        if key:
            with_key += 1
            human_with_key += human
            by_source[source or "?"] = by_source.get(source or "?", 0) + 1
    return MatchStats(
        total=len(prs),
        with_key=with_key,
        rate=round(with_key / len(prs), 4) if prs else 0.0,
        human_total=human_total,
        human_with_key=human_with_key,
        human_rate=round(human_with_key / human_total, 4) if human_total else 0.0,
        by_source=by_source,
    )
