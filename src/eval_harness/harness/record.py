from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from eval_harness.adapters.base import Usage
from eval_harness.harness.testrun import TestResult
from eval_harness.paths import results_root


class AttemptRecord(BaseModel):
    case_id: str
    run_id: str
    model: str
    status: Literal["completed", "invalid", "error"]
    phase: str = "done"
    error: str | None = None
    tests_before: TestResult | None = None
    tests_after: TestResult | None = None
    full_suite: TestResult | None = None
    regression_candidates: list[str] | None = None  # full-suite failures outside the baseline
    regressions: list[str] | None = None  # candidates that still fail when re-run in isolation
    lint_ok: bool | None = None
    lint_output: str | None = None
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0  # whole attempt: container setup, tests, agent, grading
    agent_wall_clock_seconds: float | None = None  # model loop only
    turns: int = 0
    tool_calls: int = 0
    files_touched: list[str] = Field(default_factory=list)
    touched_test_files: list[str] = Field(default_factory=list)
    generated_diff: str = ""
    cap_hit: str | None = None
    stop_reason: str = ""
    request_settings: dict[str, Any] = Field(default_factory=dict)
    tool_log: list[dict[str, Any]] = Field(default_factory=list)
    final_text: str = ""
    started_at: str = ""
    finished_at: str = ""

    # An imported case can carry a named-test criterion; ours are graded whole-file. Copied
    # onto the record at run time so a record stays self-contained once written.
    criterion: dict[str, list[str]] | None = None

    @property
    def resolved(self) -> bool:
        if self.status != "completed" or self.tests_after is None:
            return False
        if self.criterion:
            from eval_harness.collect.cases import TestCriterion

            return TestCriterion.model_validate(self.criterion).met(self.tests_after.failed_tests)
        return self.tests_after.ok

    @property
    def no_regression(self) -> bool | None:
        return None if self.regressions is None else not self.regressions


class RunMeta(BaseModel):
    run_id: str
    model: str
    dataset: str | None = None  # which named set of cases this run covered
    model_config_snapshot: dict[str, Any]
    caps: dict[str, Any]
    concurrency: int
    harness_git_sha: str
    image: str
    cases: list[str]
    started_at: str
    finished_at: str | None = None


def results_dir(run_id: str) -> Path:
    d = results_root() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_record(rec: AttemptRecord) -> Path:
    path = results_dir(rec.run_id) / f"{rec.case_id}.json"
    path.write_text(json.dumps(rec.model_dump(), indent=2) + "\n")
    return path


def load_record(run_id: str, case_id: str) -> AttemptRecord | None:
    path = results_dir(run_id) / f"{case_id}.json"
    return AttemptRecord.model_validate_json(path.read_text()) if path.exists() else None


def save_run_meta(meta: RunMeta) -> Path:
    path = results_dir(meta.run_id) / "run.json"
    path.write_text(json.dumps(meta.model_dump(), indent=2) + "\n")
    return path
