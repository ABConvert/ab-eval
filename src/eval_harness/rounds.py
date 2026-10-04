"""Private, immutable benchmark definitions with append-only model attempts."""

from __future__ import annotations

import fcntl
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from eval_harness.adapters.base import Caps
from eval_harness.collect.cases import Case, load_case, save_case
from eval_harness.collect.datasets import load as load_dataset
from eval_harness.config import ModelConfig, RepoConfig, load_models, load_repos
from eval_harness.harness.runner import harness_git_sha
from eval_harness.paths import results_root
from eval_harness.report.leaderboard import WEIGHTS


class Entry(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model: ModelConfig
    run_ids: list[str] = Field(default_factory=list)


class Round(BaseModel):
    id: str
    name: str
    dataset: str
    created_at: str
    cases: list[Case]
    repo: RepoConfig
    caps: Caps
    judge: ModelConfig
    harness_sha: str
    weights: dict[str, float]
    entries: list[Entry] = Field(default_factory=list)


def directory(round_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", round_id):
        raise ValueError("Invalid round ID")
    return results_root() / "_rounds" / round_id


def load(round_id: str) -> Round:
    return Round.model_validate_json((directory(round_id) / "round.json").read_text())


def all_rounds() -> list[Round]:
    root = results_root() / "_rounds"
    return sorted(
        [load(p.name) for p in root.iterdir() if (p / "round.json").exists()]
        if root.exists()
        else [],
        key=lambda r: r.created_at,
        reverse=True,
    )


def _save(round_: Round) -> None:
    target = directory(round_.id) / "round.json"
    temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(round_.model_dump_json(indent=2) + "\n")
    temporary.replace(target)


@contextmanager
def _locked(round_id: str) -> Iterator[None]:
    with (directory(round_id) / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def create(name: str, dataset: str, *, source: str | None = None) -> Round:
    name = name.strip()
    if not name or len(name) > 100:
        raise ValueError("Give the round a name of 1-100 characters")
    if source:
        round_ = load(source).model_copy(deep=True)
        round_.entries = []
        round_.name = name
    else:
        cases = [load_case(cid) for cid in load_dataset(dataset).case_ids]
        if not cases or len({c.repo for c in cases}) != 1:
            raise ValueError("Choose a non-empty case set from one repository")
        repo = next((r for r in load_repos().values() if r.github == cases[0].repo), None)
        if repo is None:
            raise ValueError("Configure the repository for this case set in Settings")
        models = load_models()
        if "judge" not in models:
            raise ValueError("Configure a judge in Settings before creating a round")
        round_ = Round(
            id="0" * 12,
            name=name,
            dataset=dataset,
            created_at="",
            cases=cases,
            repo=repo,
            caps=Caps(),
            judge=models["judge"],
            harness_sha=harness_git_sha(),
            weights=dict(WEIGHTS),
        )
    round_.harness_sha = harness_git_sha()
    round_.id = uuid.uuid4().hex[:12]
    round_.created_at = datetime.now(UTC).isoformat()
    directory(round_.id).mkdir(parents=True, exist_ok=False)
    for case in round_.cases:
        save_case(case, directory(round_.id) / "cases")
    _save(round_)
    return round_


def append_models(round_id: str, keys: list[str]) -> list[tuple[Entry, str]]:
    """Reserve new attempts before queueing; repeated identical configurations are refused."""
    models = load_models()
    models["human-patch"] = ModelConfig(
        key="human-patch", provider="human-patch", model="human-patch"
    )
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Select one or more different models")
    if any(k in ("judge", "classifier") or k not in models for k in keys):
        raise ValueError("Select configured candidate models; the judge is not a candidate")
    with _locked(round_id):
        round_ = load(round_id)
        if round_.harness_sha != harness_git_sha():
            raise ValueError("The evaluator changed. Create a new round to use this version")
        candidates = [models[k] for k in keys]
        if any(m == entry.model for m in candidates for entry in round_.entries):
            raise ValueError("This model configuration already belongs to the round")
        additions = []
        for model in candidates:
            entry = Entry(model=model)
            run_id = f"round-{round_id}-{entry.id}-1"
            entry.run_ids.append(run_id)
            round_.entries.append(entry)
            additions.append((entry, run_id))
        _save(round_)
        return additions


def retry(round_id: str, entry_id: str) -> tuple[Entry, str]:
    with _locked(round_id):
        round_ = load(round_id)
        if round_.harness_sha != harness_git_sha():
            raise ValueError("The evaluator changed. Create a new round to use this version")
        entry = next((e for e in round_.entries if e.id == entry_id), None)
        if entry is None:
            raise ValueError("Model not found in this round")
        run_id = f"round-{round_id}-{entry.id}-{len(entry.run_ids) + 1}"
        entry.run_ids.append(run_id)
        _save(round_)
        return entry, run_id


def execution(round_id: str, run_id: str) -> tuple[Round, Entry]:
    round_ = load(round_id)
    if round_.harness_sha != harness_git_sha():
        raise ValueError("Evaluator version differs from the frozen round")
    entry = next((e for e in round_.entries if run_id in e.run_ids), None)
    if entry is None:
        raise ValueError("Attempt does not belong to this round")
    if (results_root() / run_id / "run.json").exists():
        raise ValueError("This attempt already started; retry creates a separate attempt")
    return round_, entry
