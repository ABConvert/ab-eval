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


def load_verdicts(path: Path | None = None) -> dict[str, Verdict]:
    path = path or curation_path()
    raw = json.loads(path.read_text())
    return {key: Verdict.model_validate(v) for key, v in raw["verdicts"].items()}
