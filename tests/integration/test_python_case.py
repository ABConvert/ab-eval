"""A Python case, validated end to end in Docker, built from nothing but a temporary repository.

The other integration test needs a private demo clone and a Node suite, so no Python path
ran for real anywhere: three bugs lived there until a real repository found them — the uv
environment installed where no volume kept it, a login shell that dropped it from PATH, and a
full-suite command that never wrote the report it was read from. This builds a tiny uv
project with a bug and its fix, and needs only Docker and network for the install.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from eval_harness.adapters.base import Caps
from eval_harness.adapters.fake import NoopAdapter
from eval_harness.collect import cases
from eval_harness.config import RepoConfig
from eval_harness.harness.runner import run_many

pytestmark = pytest.mark.docker

PYPROJECT = """\
[project]
name = "calc"
version = "0"
requires-python = ">=3.12"
dependencies = []

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
pythonpath = ["."]
"""

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
OTHER = "def test_unrelated():\n    assert True\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _repo(root: Path) -> tuple[Path, str, str]:
    """A uv project: base commit with the bug, head commit with the fix and its test."""
    repo = root / "calc"
    (repo / "tests").mkdir(parents=True)
    _git(repo, "init", "-q")
    for key, value in (
        ("user.name", "Test"),
        ("user.email", "t@example.invalid"),
        ("commit.gpgSign", "false"),
    ):
        _git(repo, "config", key, value)
    (repo / "pyproject.toml").write_text(PYPROJECT)
    (repo / "calc.py").write_text(BUGGY)
    (repo / "tests" / "test_other.py").write_text(OTHER)
    subprocess.run(["uv", "lock", "-q"], cwd=repo, check=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "calc.py").write_text(FIXED)
    (repo / "tests" / "test_calc.py").write_text(TEST)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fix add")
    head = _git(repo, "rev-parse", "HEAD").strip()
    return repo, base, head


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from eval_harness import paths

    monkeypatch.setenv("ABEVAL_DATA_ROOT", str(tmp_path / "root"))
    paths.reset_cache()
    return tmp_path / "root"


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to lock the fixture project")
async def test_a_uv_case_validates_in_the_sandbox(tmp_path: Path, data_root: Path) -> None:
    repo_dir, base, head = _repo(tmp_path)
    repo = RepoConfig.model_validate(
        {
            "key": "itest-py",
            "github": "acme/calc",
            "local_path": str(repo_dir),
            "image": "abeval/itest-py:python312",
            "dockerfile": "docker/python312.Dockerfile",
            "dep_dirs": [
                {
                    "dir": ".",
                    "install": "uv sync --frozen",
                    "lock": ["pyproject.toml", "uv.lock"],
                    "target": ".venv",
                }
            ],
            "runners": {
                "pytest": {
                    "kind": "pytest",
                    "cwd": ".",
                    "match": ["**/test_*.py"],
                    "full": "python -m pytest",
                    "deps": ["."],
                }
            },
            "test_file_globs": ["**/tests/**"],
            "non_code_globs": [],
            "limits": {"cpus": 2, "memory": "2g", "pids": 1024},
        }
    )
    case = cases.Case(
        case_id="PY-1",
        repo="acme/calc",
        kind="bug_fix",
        base_commit=base,
        task_prompt="add() subtracts",
        test_command="python -m pytest tests/test_calc.py",
        test_files_added=["tests/test_calc.py"],
        test_files=["tests/test_calc.py"],
        code_files=["calc.py"],
        human_patch=_git(repo_dir, "diff", base, head, "--", "calc.py"),
        human_test_patch=_git(repo_dir, "diff", base, head, "--", "tests"),
        files_changed=2,
        lines_changed=6,
        human_cycle_time_hours=0.0,
        review_comment_count=0,
        merged_at="2026-01-01T00:00:00Z",
        pr_number=1,
        merge_commit=head,
        test_runner="pytest",
        files_dropped=[],
        tier="A",
        kind_source="review",
        author="t",
        author_kind="human",
        spec_level_heuristic="symptom",
        ticket_created_at="2026-01-01T00:00:00Z",
        ticket_started_at=None,
        human_cycle_time_source="created_at",
        test_groups=[cases.TestGroup(runner="pytest", files=["tests/test_calc.py"])],
    )

    [rec] = await run_many(
        [case],
        repo,
        adapter=NoopAdapter(),
        caps=Caps(),
        run_id="validate",
        concurrency=1,
        retry_errors=True,
        validate_only=True,
        recheck=True,
    )

    assert rec.status == "completed", f"{rec.error}\n{rec.tests_before}"
    # The fix's own test fails first: it imports fine, and add(2, 3) is -1.
    assert rec.tests_before is not None and rec.tests_before.failed == 1
    # The baseline was read, not merely run: both files, from the junit report.
    assert rec.full_suite is not None and rec.full_suite.total == 2
    assert rec.tests_after is not None and rec.tests_after.passed == 1
