"""Read-only views over data/ and results/ for the dashboard.

Every function reads files the CLI already writes; the dashboard never invents state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_harness.collect.cases import Case, load_case
from eval_harness.harness.record import AttemptRecord
from eval_harness.paths import cases_dir, results_root, splits_path
from eval_harness.report.render import difficulty_of
from eval_harness.report.summary import RunSummary

VALIDATE_RUN = "validate"
# Directories under results/ that are not model runs.
RESERVED = {VALIDATE_RUN, "_jobs", "_rate-limited"}


def title_of(case: Case) -> str:
    """First markdown heading of the task prompt, else the case id."""
    for line in case.task_prompt.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return case.case_id


def splits() -> dict[str, set[str]]:
    if not splits_path().exists():
        return {}
    raw = json.loads(splits_path().read_text())
    return {k: set(v) for k, v in raw.items() if isinstance(v, list)}


@dataclass
class CaseRow:
    case_id: str
    title: str
    kind: str
    split: str
    tier: str
    author_kind: str
    spec_level: str
    difficulty: str  # easy | medium | hard, rated from the case, never from results
    validation: str  # valid | invalid | error | pending
    validation_note: str
    baseline_failures: int | None
    last_run: str | None = None
    last_result: str | None = None


def _validation_of(rec: AttemptRecord | None) -> tuple[str, str]:
    if rec is None:
        return "pending", "not validated yet"
    if rec.status == "invalid":
        return "invalid", rec.error or "invalid"
    if rec.status == "error":
        return "error", (rec.error or "error").splitlines()[0][:120]
    before = rec.tests_before
    n = before.failed if before else 0
    return "valid", f"{n} fail before the fix"


def load_records(run_id: str) -> dict[str, AttemptRecord]:
    out: dict[str, AttemptRecord] = {}
    d = results_root() / run_id
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            rec = AttemptRecord.model_validate_json(path.read_text())
        except Exception:  # a record being written right now
            continue
        out[rec.case_id] = rec
    return out


def case_rows() -> list[CaseRow]:
    sp = splits()
    validated = load_records(VALIDATE_RUN)
    latest = _latest_attempts()
    rows: list[CaseRow] = []
    for path in sorted(cases_dir().glob("*.json")):
        case = load_case(path.stem)
        state, note = _validation_of(validated.get(case.case_id))
        run_id, result = latest.get(case.case_id, (None, None))
        rows.append(
            CaseRow(
                case_id=case.case_id,
                title=title_of(case),
                kind=case.kind,
                split=next((k for k, ids in sp.items() if case.case_id in ids), "?"),
                tier=case.tier,
                author_kind=case.author_kind,
                spec_level=case.spec_level_heuristic,
                difficulty=difficulty_of(case.case_id),
                validation=state,
                validation_note=note,
                baseline_failures=(
                    len(case.full_suite_baseline_failures)
                    if case.full_suite_baseline_failures is not None
                    else None
                ),
                last_run=run_id,
                last_result=result,
            )
        )
    rows.sort(key=lambda r: r.case_id, reverse=True)
    return rows


def _latest_attempts() -> dict[str, tuple[str, str]]:
    """{case_id: (run_id, outcome)} from the most recently started run that touched it."""
    out: dict[str, tuple[str, str]] = {}
    for meta in sorted(run_metas(), key=lambda m: m.get("started_at") or ""):
        for case_id, rec in load_records(str(meta["run_id"])).items():
            out[case_id] = (str(meta["run_id"]), outcome_of(rec))
    return out


def outcome_of(rec: AttemptRecord) -> str:
    if rec.status == "invalid":
        return "invalid"
    if rec.status == "error":
        return "error"
    if rec.resolved:
        return "resolved"
    return f"failed · {rec.cap_hit}" if rec.cap_hit else "failed"


def run_metas() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = results_root()
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in RESERVED:
            continue
        path = d / "run.json"
        if not path.exists():
            continue
        try:
            out.append(json.loads(path.read_text()))
        except Exception:
            continue
    out.sort(key=lambda m: str(m.get("started_at") or ""), reverse=True)
    return out


@dataclass
class RunView:
    run_id: str
    model: str
    started_at: str
    finished_at: str | None
    caps: dict[str, Any]
    harness_git_sha: str
    case_ids: list[str]
    records: dict[str, AttemptRecord] = field(default_factory=dict)
    summary: RunSummary | None = None

    @property
    def done(self) -> int:
        return len(self.records)

    @property
    def total(self) -> int:
        return len(self.case_ids)

    @property
    def resolved(self) -> int:
        return sum(1 for r in self.records.values() if r.resolved)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.records.values() if r.status == "completed" and not r.resolved)

    @property
    def broken(self) -> int:
        return sum(1 for r in self.records.values() if r.status != "completed")

    @property
    def cost(self) -> float:
        """What this run cost, from the records — or from the summary when they are gone.

        Records get pruned. Summing only what is on disk made /runs report $76.98 for
        runs that cost $535: six of them read $0.00 in the cost column while their own
        results page, reading the same run's summary, read $160.30 and $124.18. Two
        pages in one session disagreeing by $160, on the page you open to ask what the
        harness has cost you.
        """
        from_records = round(sum(r.cost_usd for r in self.records.values()), 4)
        if from_records or self.summary is None:
            return from_records
        return round(sum(c.cost_usd for c in self.summary.cases), 4)

    @property
    def pending_ids(self) -> list[str]:
        return [c for c in self.case_ids if c not in self.records]

    @property
    def scored(self) -> bool:
        return self.summary is not None

    def running_case(self, job_attached: bool) -> str | None:
        """The case a live job is working on: the first one with no record yet."""
        pending = self.pending_ids
        return pending[0] if job_attached and pending else None


def load_run(run_id: str) -> RunView | None:
    meta_path = results_root() / run_id / "run.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    summary_path = results_root() / run_id / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = RunSummary.model_validate_json(summary_path.read_text())
        except Exception:
            summary = None
    return RunView(
        run_id=run_id,
        model=str(meta.get("model") or "?"),
        started_at=str(meta.get("started_at") or ""),
        finished_at=meta.get("finished_at"),
        caps=dict(meta.get("caps") or {}),
        harness_git_sha=str(meta.get("harness_git_sha") or "?"),
        case_ids=[str(c) for c in (meta.get("cases") or [])],
        records=load_records(run_id),
        summary=summary,
    )


def runs() -> list[RunView]:
    out = []
    for meta in run_metas():
        view = load_run(str(meta["run_id"]))
        if view is not None:
            out.append(view)
    return out


@dataclass
class CaseAttempt:
    """One model's attempt at a case, next to its score when the run has been scored."""

    run_id: str
    model: str
    started_at: str
    record: AttemptRecord
    score: Any = None  # CaseScore | None; the run may not be scored yet

    @property
    def outcome(self) -> str:
        return outcome_of(self.record)


def case_across_runs(case_id: str) -> list[CaseAttempt]:
    """Every run that attempted this case, newest run first."""
    out: list[CaseAttempt] = []
    for view in runs():
        rec = view.records.get(case_id)
        if rec is None:
            continue
        score = None
        if view.summary is not None:
            score = next((c for c in view.summary.cases if c.case_id == case_id), None)
        out.append(
            CaseAttempt(
                run_id=view.run_id,
                model=view.model,
                started_at=view.started_at,
                record=rec,
                score=score,
            )
        )
    return out


def summaries(run_ids: list[str]) -> list[RunSummary]:
    out = []
    for rid in run_ids:
        path = results_root() / rid / "summary.json"
        if path.exists():
            out.append(RunSummary.model_validate_json(path.read_text()))
    return out


def dataset_of(view: RunView) -> str | None:
    """The dataset a run covered: what it recorded, else whatever set matches its cases."""
    from eval_harness.collect.datasets import matching

    meta = results_root() / view.run_id / "run.json"
    if meta.exists():
        try:
            recorded = json.loads(meta.read_text()).get("dataset")
            if recorded:
                return str(recorded)
        except Exception:
            pass
    return matching(view.case_ids)


def comparison_files() -> list[Path]:
    root = results_root()
    return sorted(root.glob("compare-*.md")) if root.is_dir() else []


def dataset_case_ids(name: str) -> list[str]:
    """Case ids in a named dataset; an empty list when it cannot be read."""
    from eval_harness.collect.datasets import load as load_dataset

    try:
        return list(load_dataset(name).case_ids)
    except Exception:
        return []
