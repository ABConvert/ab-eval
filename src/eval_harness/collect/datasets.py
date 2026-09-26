"""Named sets of cases.

A dataset is just an ordered list of case ids with a name, so the same case can sit in
several of them. `dev` and `holdout` from data/splits.json are exposed as built-in
datasets so older runs still name the data they used.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from eval_harness.paths import cases_dir, datasets_dir, splits_path

BUILT_IN = ("all", "dev", "holdout")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Dataset(BaseModel):
    name: str
    description: str = ""
    created_at: str = Field(default_factory=_now)
    case_ids: list[str] = Field(default_factory=list)

    @property
    def built_in(self) -> bool:
        return self.name in BUILT_IN

    def __len__(self) -> int:
        return len(self.case_ids)

    def __bool__(self) -> bool:
        # Without this, __len__ makes a new empty dataset falsy and templates that say
        # "if dataset" silently skip it.
        return True


def validate_name(name: str) -> str:
    name = name.strip().lower()
    if not NAME_RE.match(name):
        raise ValueError(
            "dataset names are lowercase letters, digits, dot, dash or underscore, "
            f"up to 64 characters; got {name!r}"
        )
    if name in BUILT_IN:
        raise ValueError(f"{name!r} is a built-in dataset and cannot be overwritten")
    return name


def path_for(name: str) -> Path:
    return datasets_dir() / f"{name}.json"


def all_case_ids() -> list[str]:
    return sorted(p.stem for p in cases_dir().glob("*.json"))


def _split_ids(which: str) -> list[str]:
    if not splits_path().exists():
        return []
    raw = json.loads(splits_path().read_text())
    ids = raw.get(which)
    return sorted(ids) if isinstance(ids, list) else []


def load(name: str) -> Dataset:
    if name == "all":
        return Dataset(name="all", description="Every collected case", case_ids=all_case_ids())
    if name in ("dev", "holdout"):
        return Dataset(
            name=name,
            description=f"The {name} split from data/splits.json",
            case_ids=_split_ids(name),
        )
    path = path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"no dataset {name!r}")
    return Dataset.model_validate_json(path.read_text())


def save(ds: Dataset) -> Path:
    validate_name(ds.name)
    datasets_dir().mkdir(parents=True, exist_ok=True)
    path = path_for(ds.name)
    path.write_text(json.dumps(ds.model_dump(), indent=2) + "\n")
    return path


def create(name: str, description: str = "", case_ids: list[str] | None = None) -> Dataset:
    name = validate_name(name)
    if path_for(name).exists():
        raise FileExistsError(f"dataset {name!r} already exists")
    ds = Dataset(name=name, description=description, case_ids=list(case_ids or []))
    save(ds)
    return ds


def delete(name: str) -> None:
    validate_name(name)
    path_for(name).unlink(missing_ok=True)


def add_cases(name: str, case_ids: list[str]) -> Dataset:
    """Append cases, keeping order and dropping duplicates."""
    ds = load(name)
    if ds.built_in:
        raise ValueError(f"{name!r} is derived from the case files and cannot be edited")
    known = set(all_case_ids())
    unknown = [c for c in case_ids if c not in known]
    if unknown:
        raise LookupError(f"no case file for {', '.join(unknown)}; collect them first")
    seen = set(ds.case_ids)
    ds.case_ids += [c for c in case_ids if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
    save(ds)
    return ds


def remove_cases(name: str, case_ids: list[str]) -> Dataset:
    ds = load(name)
    if ds.built_in:
        raise ValueError(f"{name!r} is derived from the case files and cannot be edited")
    drop = set(case_ids)
    ds.case_ids = [c for c in ds.case_ids if c not in drop]
    save(ds)
    return ds


def names() -> list[str]:
    saved = sorted(p.stem for p in datasets_dir().glob("*.json")) if datasets_dir().is_dir() else []
    return [*BUILT_IN, *saved]


def load_all() -> list[Dataset]:
    out = []
    for n in names():
        try:
            out.append(load(n))
        except Exception:  # a hand-edited file should not break the listing
            continue
    return out


def containing(case_id: str) -> list[str]:
    """Names of every dataset that holds this case."""
    return [ds.name for ds in load_all() if case_id in ds.case_ids]


def matching(case_ids: list[str]) -> str | None:
    """The dataset whose contents are exactly these cases, preferring a saved one."""
    want = set(case_ids)
    saved = [ds for ds in load_all() if not ds.built_in]
    for ds in [*saved, *(load(n) for n in BUILT_IN)]:
        if set(ds.case_ids) == want and want:
            return ds.name
    return None
