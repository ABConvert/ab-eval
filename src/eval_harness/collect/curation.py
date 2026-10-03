from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from eval_harness.paths import curation_path


class Verdict(BaseModel):
    tier: str
    reason: str
    pr: int
    author: str = ""
    author_kind: str = "human"
    kind_review: str = "-"
    spec_level_heuristic: str = "symptom"


class Candidate(BaseModel):
    """A PR that passed the structural rules and is waiting for a person to tier it."""

    key: str
    pr: int
    title: str
    author: str
    author_kind: str = "human"
    files: int
    lines: int


def load_verdicts(path: Path | None = None) -> dict[str, Verdict]:
    """Verdicts by ticket key; none yet on a fresh data root, which is not an error."""
    path = path or curation_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {key: Verdict.model_validate(v) for key, v in raw.get("verdicts", {}).items()}


def save_verdicts(verdicts: dict[str, Verdict], path: Path | None = None) -> Path:
    path = path or curation_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"verdicts": {k: v.model_dump() for k, v in sorted(verdicts.items())}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    tmp.replace(path)
    return path


def pending_path(repo_key: str) -> Path:
    return curation_path().parent / f"pending-{repo_key}.json"


def save_pending(repo_key: str, candidates: list[Candidate]) -> Path:
    """What the last collect left for review. Replaced on every collect: it is a work list."""
    path = pending_path(repo_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.model_dump() for c in candidates], indent=1) + "\n")
    return path


def load_pending(repo_key: str) -> list[Candidate]:
    path = pending_path(repo_key)
    if not path.exists():
        return []
    return [Candidate.model_validate(c) for c in json.loads(path.read_text())]


def record_verdicts(
    repo_key: str,
    keys: list[str],
    *,
    tier: str,
    reason: str,
    kind: str = "-",
    spec_level: str = "symptom",
    path: Path | None = None,
) -> list[str]:
    """Write a verdict for each key from the pending list. Returns keys that were not pending."""
    pending = {c.key: c for c in load_pending(repo_key)}
    verdicts = load_verdicts(path)
    unknown: list[str] = []
    for key in keys:
        c = pending.get(key.upper())
        if c is None:
            unknown.append(key)
            continue
        verdicts[c.key] = Verdict(
            tier=tier,
            reason=reason,
            pr=c.pr,
            author=c.author,
            author_kind=c.author_kind,
            kind_review=kind,
            spec_level_heuristic=spec_level,
        )
    save_verdicts(verdicts, path)
    return unknown
