"""Tests configure against a fixture repository, never the operator's own.

A test that reads `config/repos.yaml` fails on a fresh clone, where there is no such file, and
passes for the wrong reason on a configured one. Pointing the data root at `tests/fixtures`
gives every test the same repository whatever the machine holds.

This runs at import, not in a fixture: some test modules call `load_repos()` at module level,
which happens before any fixture could have set the variable.
"""

from __future__ import annotations

import os
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
os.environ.setdefault("ABEVAL_DATA_ROOT", str(FIXTURES))

from eval_harness import paths  # noqa: E402  - must follow the environment it depends on

paths.reset_cache()


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_data_root() -> object:
    """Reset the resolved root after every test.

    `data_root()` is cached, and a test that relocates it leaves that cache pointing at a
    temporary directory that no longer exists — which the next test then inherits. monkeypatch
    restores the environment variable but knows nothing about the cache.
    """
    yield
    paths.reset_cache()


@pytest.fixture
def demo_repo() -> object:
    """The fixture repository, loaded by path so it survives a relocated data root."""
    from eval_harness.config import load_repos

    return load_repos(FIXTURES / "config" / "repos.yaml")["demo-app"]
