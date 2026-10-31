"""Prompt cases: a frozen (system, prompt) pair plus what only the graders may see."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from eval_harness.paths import prompt_cases_dir
from eval_harness.prompt.task import load_task_file

# A case id becomes a file name under results/, so it is held to what is safe there.
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PromptCase(BaseModel):
    # Forbid, because a producer that misspells `grading_context` would otherwise hand the
    # judges nothing and every fidelity score would quietly measure the empty dict.
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: str
    kind: Literal["prompt"] = "prompt"
    system: str
    prompt: str
    prompt_version: str = ""
    max_output_tokens: int | None = None
    # Shown to judges and scorers only, never to the model under test.
    grading_context: dict[str, Any] = Field(default_factory=dict)
    strata: list[str] = Field(default_factory=list)
    split: Literal["dev", "holdout"] | None = None
    source: str = ""
    created_at: str = ""

    @field_validator("case_id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        if not CASE_ID_RE.match(v) or ".." in v:
            raise ValueError(f"case_id must be letters, digits, . _ - (a file name); got {v!r}")
        return v


def case_path(task: str, case_id: str) -> Path:
    return prompt_cases_dir(task) / f"{case_id}.json"


def save_prompt_case(case: PromptCase) -> Path:
    path = case_path(case.task, case.case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case.model_dump(), indent=2, ensure_ascii=False) + "\n")
    return path


def load_prompt_case(task: str, case_id: str) -> PromptCase:
    return PromptCase.model_validate_json(case_path(task, case_id).read_text())


def prompt_case_ids(task: str) -> list[str]:
    d = prompt_cases_dir(task)
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def load_prompt_cases(task: str, ids: list[str] | None = None) -> list[PromptCase]:
    return [load_prompt_case(task, i) for i in (ids if ids is not None else prompt_case_ids(task))]


def all_prompt_case_ids() -> set[str]:
    root = prompt_cases_dir()
    if not root.is_dir():
        return set()
    return {p.stem for p in root.glob("*/*.json")}


@dataclass
class ImportReport:
    task: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    splits_assigned: int = 0
    config_copied: list[str] = field(default_factory=list)


class ImportRejected(ValueError):
    """Nothing was imported; `problems` says why, one line per file."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


def import_prompt_cases(
    task: str, source: Path, *, holdout_fraction: float | None = None
) -> ImportReport:
    """Validate every case in `source`, then copy them and the task's config in.

    All or nothing: one malformed file rejects the import, because a dataset that is
    silently missing the cases the producer could not serialise is a biased dataset.
    """
    from eval_harness.collect.split import assign_splits

    if not source.is_dir():
        raise ImportRejected([f"{source} is not a directory"])
    problems: list[str] = []
    cases: list[PromptCase] = []
    for path in sorted(source.glob("*.json")):
        try:
            case = PromptCase.model_validate_json(path.read_text())
        except (ValidationError, ValueError) as e:
            problems.append(f"{path.name}: {str(e).splitlines()[0]}" + _first_error(e))
            continue
        if case.task != task:
            problems.append(f"{path.name}: task is {case.task!r}, not {task!r}")
            continue
        if path.stem != case.case_id:
            problems.append(f"{path.name}: file name does not match case_id {case.case_id!r}")
            continue
        cases.append(case)
    if not cases and not problems:
        problems.append(f"{source} holds no *.json case files")

    target = prompt_cases_dir(task)
    staged_task = source / "task.yaml"
    if staged_task.exists():
        try:
            cfg = load_task_file(staged_task)
            if cfg.task != task:
                problems.append(f"task.yaml: task is {cfg.task!r}, not {task!r}")
            else:
                cfg.load_rubric()
        except Exception as e:
            problems.append(f"task.yaml: {type(e).__name__}: {str(e).splitlines()[0]}")
    elif not (target / "task.yaml").exists():
        problems.append(f"no task.yaml in {source} and none already imported for {task!r}")
    if problems:
        raise ImportRejected(problems)

    report = ImportReport(task=task)
    if holdout_fraction is not None:
        unsplit = [c for c in cases if c.split is None]
        assigned = assign_splits(unsplit, holdout_fraction)
        for c in unsplit:
            c.split = assigned[c.case_id]  # type: ignore[assignment]
        report.splits_assigned = len(unsplit)
    target.mkdir(parents=True, exist_ok=True)
    if staged_task.exists():
        cfg = load_task_file(staged_task)
        for name in ("task.yaml", cfg.rubric):
            shutil.copyfile(source / name, target / name)
            report.config_copied.append(name)
    for case in cases:
        path = case_path(task, case.case_id)
        before = path.read_text() if path.exists() else None
        save_prompt_case(case)
        after = path.read_text()
        if before is None:
            report.added.append(case.case_id)
        elif before == after:
            report.unchanged.append(case.case_id)
        else:
            report.updated.append(case.case_id)
    return report


def _first_error(e: Exception) -> str:
    if isinstance(e, ValidationError) and e.errors():
        err = e.errors()[0]
        return f" ({'.'.join(str(x) for x in err['loc'])}: {err['msg']})"
    return ""
