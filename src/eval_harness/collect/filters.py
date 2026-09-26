from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel

from eval_harness.collect.github import PullRequest
from eval_harness.collect.join import ticket_for
from eval_harness.collect.linear import Issue
from eval_harness.config import RepoConfig

MIN_DESCRIPTION = 200
MAX_FILES = 10
MAX_LINES = 400
MIN_TEST_LINES = 10
MIN_CODE_LINES = 8
REVERT_WINDOW_DAYS = 30

DEP_FILES = re.compile(
    r"(^|/)(package(-lock)?\.json|yarn\.lock|pnpm-lock\.yaml|pubspec\.(yaml|lock)|uv\.lock|poetry\.lock|requirements[^/]*\.txt|Cargo\.lock)$"
)
GENERATED = re.compile(r"(^|/)(generated|__generated__|dist|build)/|\.(min\.js|snap)$")
CONFIG_ONLY = re.compile(
    r"\.(json|ya?ml|toml|ini|env|cfg|conf|lock)$|(^|/)\.[^/]+rc$|(^|/)Dockerfile$"
)


class Decision(BaseModel):
    pr: int
    key: str | None
    accept: bool
    rule: str | None = None
    detail: str = ""


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def reverted_numbers(prs: list[PullRequest]) -> set[int]:
    """PR numbers referenced by a 'revert' PR merged within 30 days of them."""
    by_number = {p.number: p for p in prs}
    out: set[int] = set()
    for p in prs:
        if not re.search(r"\brevert", p.title, re.I):
            continue
        for m in re.finditer(r"#(\d{2,6})", f"{p.title} {p.body}"):
            n = int(m.group(1))
            target = by_number.get(n)
            if (
                target
                and 0 <= (_ts(p.merged_at) - _ts(target.merged_at)).days <= REVERT_WINDOW_DAYS
            ):
                out.add(n)
    return out


def decide(
    pr: PullRequest,
    issue: Issue | None,
    *,
    repo: RepoConfig,
    prs_for_key: int,
    reverted: set[int],
) -> Decision:
    key, source = ticket_for(pr, repo.linear_team)

    def rej(rule: str, detail: str = "") -> Decision:
        return Decision(pr=pr.number, key=key, accept=False, rule=rule, detail=detail)

    if pr.number in reverted:
        return rej("S1", "reverted within 30 days")
    if key is None:
        return rej("S2", "no ticket key")
    if issue is None:
        return rej("S2", f"{key} not found in Linear")
    if len(issue.description) < MIN_DESCRIPTION:
        return rej("S2", f"description {len(issue.description)} chars")
    if prs_for_key != 1:
        return rej("S6", f"{prs_for_key} merged PRs for {key}")
    if issue.children:
        return rej("R1", f"ticket has {len(issue.children)} sub-issues")
    title_keys = {
        k.upper() for k in re.findall(rf"\b{re.escape(repo.linear_team)}-\d+\b", pr.title, re.I)
    }
    if len(title_keys) > 1:
        return rej("R2", f"title names {sorted(title_keys)}")
    branch_key = re.search(rf"\b{re.escape(repo.linear_team)}-\d+\b", pr.head_ref, re.I)
    if branch_key and title_keys and branch_key.group(0).upper() not in title_keys:
        return rej("R3", f"branch {branch_key.group(0)} vs title {sorted(title_keys)}")
    if _ts(issue.created_at) > _ts(pr.created_at):
        return rej("R4", "ticket created after PR opened")
    tests = [f for f in pr.files if repo.is_test_file(f.path)]
    code = [f for f in pr.files if repo.is_code_file(f.path)]
    if not tests or not code:
        return rej("S3", f"tests={len(tests)} code={len(code)}")
    lines = sum(f.additions + f.deletions for f in tests + code)
    if len(tests) + len(code) > MAX_FILES or lines > MAX_LINES:
        return rej("S4", f"{len(tests) + len(code)} files, {lines} lines")
    paths = [f.path for f in pr.files]
    if all(DEP_FILES.search(p) for p in paths):
        return rej("S5", "dependency bump")
    if all(GENERATED.search(p) for p in paths):
        return rej("S5", "generated files only")
    if all(CONFIG_ONLY.search(p) for p in paths):
        return rej("S5", "config only")
    test_lines = sum(f.additions + f.deletions for f in tests)
    code_lines = sum(f.additions + f.deletions for f in code)
    if test_lines < MIN_TEST_LINES:
        return rej("R7", f"{test_lines} test lines")
    if code_lines < MIN_CODE_LINES:
        return rej("R8", f"{code_lines} code lines")
    return Decision(pr=pr.number, key=key, accept=True, detail=f"via {source}")
