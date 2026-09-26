from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from eval_harness.collect.cases import Case
from eval_harness.paths import splits_path


def assign_splits(cases: list[Case], holdout_fraction: float = 0.3) -> dict[str, str]:
    """Date-ordered split: the most recent `holdout_fraction` of cases is the holdout."""
    ordered = sorted(cases, key=lambda c: (c.merged_at, c.case_id))
    n_holdout = round(len(ordered) * holdout_fraction)
    cut = len(ordered) - n_holdout
    return {c.case_id: ("dev" if i < cut else "holdout") for i, c in enumerate(ordered)}


def write_splits(assignments: dict[str, str], path: Path | None = None) -> Path:
    path = path or splits_path()
    body = {
        "dev": sorted(k for k, v in assignments.items() if v == "dev"),
        "holdout": sorted(k for k, v in assignments.items() if v == "holdout"),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(body, indent=2) + "\n")
    return path


def load_splits(path: Path | None = None) -> dict[str, list[str]]:
    path = path or splits_path()
    raw = json.loads(path.read_text())
    return {"dev": list(raw["dev"]), "holdout": list(raw["holdout"])}
