from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime

import httpx

from eval_harness.collect.cases import Case, TestGroup, added_files, count_changed_lines, split_diff
from eval_harness.collect.github import PullRequest, _from_gh
from eval_harness.collect.linear import Issue, _from_linear, api_key
from eval_harness.collect.sanitize import sanitize_text
from eval_harness.config import RepoConfig
from eval_harness.harness.testrun import render_test_command

# Accounts that are bots or agents rather than people. Whether a PR was written by a human
# matters when a benchmark compares models against "what humans shipped", so a repo declares
# its own in repos.yaml; there is no universal list.
AGENT_AUTHORS: dict[str, str] = {}

LINEAR_URL = "https://api.linear.app/graphql"


def ticket_key_for_pr(branch: str, title: str, body: str, team: str) -> str | None:
    """Branch name first, then PR title, then PR body."""
    pattern = re.compile(rf"\b{re.escape(team)}-(\d+)\b", re.I)
    for text in (branch, title, body):
        m = pattern.search(text or "")
        if m:
            return f"{team}-{m.group(1)}"
    return None


def _gh_pr(github: str, pr_number: int) -> PullRequest:
    fields = (
        "number,title,body,headRefName,author,mergeCommit,mergedAt,createdAt,reviews,comments,files"
    )
    out = subprocess.run(
        ["gh", "pr", "view", str(pr_number), "--repo", github, "--json", fields],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return _from_gh(json.loads(out))


def _git(repo: RepoConfig, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo.local_path), *args], check=True, capture_output=True, text=True
    ).stdout


def _linear_issue(key: str) -> Issue:
    team, number = key.split("-")
    query = """
    query($team: String!, $number: Float!) {
      issues(filter: { team: { key: { eq: $team } }, number: { eq: $number } }) {
        nodes { number identifier title description createdAt startedAt completedAt
                labels { nodes { name } } comments { nodes { body } }
                children { nodes { identifier } } }
      }
    }"""
    resp = httpx.post(
        LINEAR_URL,
        headers={"Authorization": api_key(), "Content-Type": "application/json"},
        json={"query": query, "variables": {"team": team, "number": float(number)}},
        timeout=30,
    )
    resp.raise_for_status()
    nodes = resp.json()["data"]["issues"]["nodes"]
    if not nodes:
        raise LookupError(f"Linear issue {key} not found")
    return _from_linear(nodes[0])


def spec_level(description: str) -> str:
    if description.count("```") >= 4:
        return "code-in-ticket"
    solution = (
        r"^#+\s*.*(fix|change|implementation|solution|proposed|approach|scope|"
        r"summary|goal|what|delivery|plan)"
    )
    if re.search(solution, description, re.I | re.M):
        return "solution-specified"
    if re.search(r"^#+\s*.*(cause|why)", description, re.I | re.M):
        return "root-cause"
    return "symptom"


def _hours(start: str, end: str) -> float:
    a = datetime.fromisoformat(start.replace("Z", "+00:00"))
    b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return round((b - a).total_seconds() / 3600, 2)


def build_task_prompt(issue: Issue) -> str:
    parts = [f"# {issue.title}", "", issue.description]
    repro = [c for c in issue.comments if re.search(r"repro|steps to|reproduce", c, re.I)]
    if repro:
        parts += ["", "## Reproduction notes", *repro]
    return sanitize_text("\n".join(parts).strip())


def case_from(
    repo: RepoConfig,
    pr: PullRequest,
    issue: Issue,
    *,
    tier: str,
    kind: str,
    kind_source: str,
    author_kind: str | None = None,
    spec_level_heuristic: str | None = None,
) -> Case:
    """Build a Case from an already-fetched PR and issue; diffs come from the local clone."""
    merge = pr.merge_commit
    base = _git(repo, "rev-parse", f"{merge}^").strip()
    diff = _git(repo, "diff", base, merge)
    code_patch, test_patch, code_files, test_files, dropped = split_diff(
        diff, repo.is_test_file, repo.is_code_file
    )
    if not test_files or not code_files:
        raise ValueError(
            f"PR #{pr.number}: needs both code and test changes "
            f"(code={code_files}, tests={test_files})"
        )
    groups = repo.runner_groups(test_files)
    started = issue.started_at
    cycle_start, cycle_source = (
        (started, "started_at") if started else (issue.created_at, "created_at")
    )
    return Case(
        case_id=issue.key,
        repo=repo.github,
        kind=kind,
        base_commit=base,
        task_prompt=build_task_prompt(issue),
        test_command=" && ".join(render_test_command(r, fs) for _, r, fs in groups),
        test_files_added=added_files(test_patch),
        test_files=test_files,
        code_files=code_files,
        human_patch=code_patch,
        human_test_patch=test_patch,
        files_changed=len(code_files) + len(test_files),
        lines_changed=count_changed_lines(code_patch) + count_changed_lines(test_patch),
        human_cycle_time_hours=_hours(cycle_start, pr.merged_at),
        review_comment_count=pr.review_count + pr.comment_count,
        merged_at=pr.merged_at,
        pr_number=pr.number,
        merge_commit=merge,
        test_runner=groups[0][0],
        test_groups=[TestGroup(runner=n, files=fs) for n, _, fs in groups],
        files_dropped=dropped,
        tier=tier,
        kind_source=kind_source,
        author=pr.author,
        author_kind=author_kind
        or repo.agent_authors.get(pr.author, AGENT_AUTHORS.get(pr.author, "human")),
        spec_level_heuristic=spec_level_heuristic or spec_level(issue.description),
        ticket_created_at=issue.created_at,
        ticket_started_at=started,
        human_cycle_time_source=cycle_source,
    )


def build_case(repo: RepoConfig, pr_number: int, *, tier: str, kind: str, kind_source: str) -> Case:
    """Step 0 path: fetch one PR and its ticket, then build the case."""
    pr = _gh_pr(repo.github, pr_number)
    key = ticket_key_for_pr(pr.head_ref, pr.title, pr.body, repo.linear_team)
    if key is None:
        raise LookupError(f"PR #{pr_number} has no {repo.linear_team} key")
    return case_from(repo, pr, _linear_issue(key), tier=tier, kind=kind, kind_source=kind_source)
