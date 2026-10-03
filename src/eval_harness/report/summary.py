from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval_harness.harness.record import results_dir


class CaseScore(BaseModel):
    case_id: str
    kind: str
    tier: str
    author_kind: str
    spec_level: str
    split: str
    status: str
    resolved: bool
    tests_pass: float
    no_regression: float | None
    lint_clean: float | None
    patch_similarity: float | None
    code_quality: float | None
    code_quality_reason: str | None = None
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    wall_clock_seconds: float
    agent_wall_clock_seconds: float | None
    turns: int
    tool_calls: int
    cap_hit: str | None
    touched_test_files: bool
    host_exec_items: int
    files_touched: int
    diff_lines: int


class RunSummary(BaseModel):
    run_id: str
    model: str
    model_config_snapshot: dict[str, Any]
    # The budget the run actually got, after the model's `caps:` block and any flag that
    # overrode it. Without this a published number cannot be reproduced from its own
    # artifact: in bench-v2 one wall clock capped the slowest model 27 times and the
    # fastest none, and the summary said nothing about which clock that was. Defaults to
    # empty so a summary written before this field keeps loading; re-score to fill it,
    # because run.json has carried the caps all along.
    caps: dict[str, Any] = Field(default_factory=dict)
    harness_git_sha: str
    judge: str | None
    judge_config: dict[str, Any] | None
    scored_at: str
    cases: list[CaseScore]
    case_difficulties: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_ids(self) -> set[str]:
        return {c.case_id for c in self.cases if c.resolved}


def summary_path(run_id: str) -> Path:
    return results_dir(run_id) / "summary.json"


def save_summary(s: RunSummary) -> Path:
    path = summary_path(s.run_id)
    path.write_text(json.dumps(s.model_dump(), indent=2) + "\n")
    return path


def load_summary(run_id: str) -> RunSummary:
    return RunSummary.model_validate_json(summary_path(run_id).read_text())
