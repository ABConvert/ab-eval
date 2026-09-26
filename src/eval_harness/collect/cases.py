from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from eval_harness.paths import cases_dir


class TestGroup(BaseModel):
    runner: str
    files: list[str]


class TestCriterion(BaseModel):
    """Which named tests decide a case, where whole-file passing is the wrong question.

    Our own cases run a file and require every test in it to pass. Imported benchmarks state it
    differently: *these* tests must go from failing to passing while *those* keep passing, and a
    pre-existing unrelated failure elsewhere in the file is not the model's problem. A case
    without one is graded the original way.
    """

    __test__ = False  # pydantic model, not a pytest class, despite the name

    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)

    def met(self, failed_after: list[str]) -> bool:
        failed = set(failed_after)
        return not (failed & set(self.fail_to_pass)) and not (failed & set(self.pass_to_pass))


class Case(BaseModel):
    case_id: str
    repo: str
    kind: str
    base_commit: str
    task_prompt: str
    test_command: str
    test_files_added: list[str]
    test_files: list[str]
    code_files: list[str]
    human_patch: str
    human_test_patch: str
    files_changed: int
    lines_changed: int
    human_cycle_time_hours: float
    review_comment_count: int
    merged_at: str
    pr_number: int
    merge_commit: str
    test_runner: str
    files_dropped: list[str]
    tier: str
    kind_source: str
    author: str
    author_kind: str
    spec_level_heuristic: str
    ticket_created_at: str
    ticket_started_at: str | None
    human_cycle_time_source: str
    full_suite_baseline_failures: list[str] | None = None
    criterion: TestCriterion | None = None  # named-test grading, for imported sets
    source: str = "collect"  # or the importer that produced it, e.g. "swebench"
    test_groups: list[TestGroup] = Field(default_factory=list)  # runner -> its test files


_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)


def _file_chunks(diff: str) -> list[tuple[str, str]]:
    """Return (path, chunk_text) for each file section of a unified diff."""
    starts = [m.start() for m in _FILE_RE.finditer(diff)]
    chunks: list[tuple[str, str]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(diff)
        chunk = diff[start:end]
        m = _FILE_RE.match(chunk)
        assert m is not None
        chunks.append((m.group(2), chunk))
    return chunks


def split_diff(
    diff: str, is_test: Callable[[str], bool], is_code: Callable[[str], bool]
) -> tuple[str, str, list[str], list[str], list[str]]:
    """Split a unified diff into (code_patch, test_patch, code_files, test_files, dropped)."""
    code_parts: list[str] = []
    test_parts: list[str] = []
    code_files: list[str] = []
    test_files: list[str] = []
    dropped: list[str] = []
    for path, chunk in _file_chunks(diff):
        if is_test(path):
            test_parts.append(chunk)
            test_files.append(path)
        elif is_code(path):
            code_parts.append(chunk)
            code_files.append(path)
        else:
            dropped.append(path)
    return "".join(code_parts), "".join(test_parts), code_files, test_files, dropped


def count_changed_lines(diff: str) -> int:
    n = 0
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            n += 1
    return n


def added_files(diff: str) -> list[str]:
    return [path for path, chunk in _file_chunks(diff) if "\nnew file mode" in chunk]


def save_case(case: Case, root: Path | None = None) -> Path:
    root = root or cases_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{case.case_id}.json"
    path.write_text(json.dumps(case.model_dump(), indent=2, ensure_ascii=False) + "\n")
    return path


def load_case(case_id: str, root: Path | None = None) -> Case:
    root = root or cases_dir()
    return Case.model_validate_json((root / f"{case_id}.json").read_text())
