"""Where the harness reads cases and writes results.

The harness and the dataset are different things with different owners: the code is shared,
the cases are a team's own merged work. `ABEVAL_DATA_ROOT` moves the whole data tree somewhere
else — another directory, another repository — without touching the checkout. It defaults to
the checkout, so an existing clone keeps the layout it already has.

Every accessor is a function rather than a module constant. A constant read at import time
would be fixed before a caller could set the variable, which is the classic version of this
bug; `data_root()` is cached once instead, on first use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from eval_harness import PROJECT_ROOT

ENV_VAR = "ABEVAL_DATA_ROOT"


@lru_cache(maxsize=1)
def data_root() -> Path:
    """The tree holding cases, datasets, curation and results."""
    raw = os.environ.get(ENV_VAR)
    return Path(raw).expanduser().resolve() if raw else PROJECT_ROOT


def reset_cache() -> None:
    """Forget the resolved root. For tests that relocate it mid-process."""
    data_root.cache_clear()


def cases_dir() -> Path:
    return data_root() / "data" / "cases"


def datasets_dir() -> Path:
    return data_root() / "data" / "datasets"


def curation_path() -> Path:
    return data_root() / "data" / "curation" / "verdicts.json"


def splits_path() -> Path:
    return data_root() / "data" / "splits.json"


def reviews_dir() -> Path:
    return data_root() / "data" / "case-reviews"


def collect_report_path() -> Path:
    return data_root() / "data" / "collect-report.json"


def raw_cache(repo_key: str) -> Path:
    """Cached GitHub and ticket payloads for one repo, so collect can re-run offline."""
    return data_root() / "data" / "raw" / repo_key


def results_root() -> Path:
    return data_root() / "results"


def jobs_dir() -> Path:
    return results_root() / "_jobs"


def config_dir() -> Path:
    return data_root() / "config"


def config_file(name: str) -> Path:
    """A config file from the data root, falling back to the checkout's own.

    `init` writes into the data root, because a team's repos.yaml belongs beside their cases.
    The checkout ships defaults — models.yaml especially — so a data root that has not
    overridden one still finds it.
    """
    theirs = config_dir() / name
    return theirs if theirs.exists() else PROJECT_ROOT / "config" / name
