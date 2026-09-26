"""Import SWE-bench instances as cases.

Two reasons to want this. A fresh clone of the harness has no cases at all, and `import swebench
--limit 5` gives someone a real benchmark in one command. And our own resolve rates cannot be
compared with anything outside themselves — this is the only external check available on an
instrument we built ourselves.

Fetched, never vendored. Each row carries a patch derived from its original repository under
that repository's licence — BSD, Apache-2.0 and in pylint's case GPL — so committing the rows
would drag a licence patchwork into this one for no benefit.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from eval_harness.collect.cases import Case, TestCriterion

ROWS_URL = "https://datasets-server.huggingface.co/rows"
VARIANTS = {
    "full": "SWE-bench/SWE-bench",
    "verified": "SWE-bench/SWE-bench_Verified",
    "lite": "SWE-bench/SWE-bench_Lite",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
}
PAGE = 100  # the datasets server's maximum


def fetch_rows(variant: str, limit: int, *, split: str = "test") -> list[dict[str, Any]]:
    """Rows straight from the datasets server. No `datasets` dependency, just REST."""
    dataset = VARIANTS[variant]
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=120) as client:
        while len(rows) < limit:
            reply = client.get(
                ROWS_URL,
                params={
                    "dataset": dataset,
                    "config": "default",
                    "split": split,
                    "offset": len(rows),
                    "length": min(PAGE, limit - len(rows)),
                },
            )
            if reply.status_code >= 400:
                raise RuntimeError(f"{dataset}: {reply.status_code} {reply.text[:200]}")
            page = [r["row"] for r in reply.json().get("rows", [])]
            if not page:
                break
            rows.extend(page)
    return rows[:limit]


def _names(raw: Any) -> list[str]:
    """FAIL_TO_PASS and PASS_TO_PASS arrive as a JSON-encoded list, or already decoded."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        return [str(x) for x in json.loads(raw or "[]")]
    except (json.JSONDecodeError, TypeError):
        return []


def to_case(row: dict[str, Any], *, variant: str) -> Case:
    """One SWE-bench instance as a Case.

    Two fields have no counterpart and are not invented. `environment_setup_commit` pins a
    per-instance environment, which repos.yaml cannot express yet, so it travels in the prompt
    where a human running this will see it. And FAIL_TO_PASS becomes a named-test criterion
    rather than a file list, because that is how SWE-bench decides a case.
    """
    fail_to_pass = _names(row.get("FAIL_TO_PASS"))
    pass_to_pass = _names(row.get("PASS_TO_PASS"))
    files = sorted({n.split("::")[0] for n in fail_to_pass + pass_to_pass if "::" in n})
    return Case(
        case_id=str(row["instance_id"]),
        repo=str(row.get("repo", "")),
        kind="bug_fix",
        base_commit=str(row.get("base_commit", "")),
        task_prompt=str(row.get("problem_statement") or ""),
        test_command="python -m pytest",
        test_files_added=[],
        test_files=files,
        code_files=[],
        human_patch=str(row.get("patch") or ""),
        human_test_patch=str(row.get("test_patch") or ""),
        files_changed=0,
        lines_changed=len((row.get("patch") or "").splitlines()),
        human_cycle_time_hours=0.0,
        review_comment_count=0,
        merged_at=str(row.get("created_at") or ""),
        pr_number=0,
        merge_commit="",
        test_runner="pytest",
        files_dropped=[],
        tier="A",
        kind_source="swebench",
        author="",
        author_kind="human",
        # SWE-bench problem statements are real issue reports, not implementation notes — the
        # opposite of the tickets a team writes for itself, which is part of why it is a useful
        # calibration set.
        spec_level_heuristic="symptom",
        ticket_created_at=str(row.get("created_at") or ""),
        ticket_started_at=None,
        human_cycle_time_source="unknown",
        criterion=TestCriterion(fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass),
        source=f"swebench:{variant}",
    )


def environment_commit(row: dict[str, Any]) -> str:
    return str(row.get("environment_setup_commit") or "")
