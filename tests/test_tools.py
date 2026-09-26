from typing import Any

import pytest

from eval_harness.adapters.base import Caps
from eval_harness.harness.sandbox import ExecResult
from eval_harness.harness.tools import TOOL_SPECS, SandboxToolExecutor


class FakeDocker:
    def __init__(self) -> None:
        self.files: dict[str, str] = {"/app/web/a.ts": "const x = 1;\nconst y = 2;\n"}
        self.commands: list[str] = []

    def exec(
        self,
        cid: str,
        cmd: Any,
        *,
        cwd: str = "/app",
        env: Any = None,
        timeout: int = 300,
        user: Any = None,
    ) -> ExecResult:
        text = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.commands.append(text)
        return ExecResult(0, "ran: " + text, "")

    def read_file(self, cid: str, path: str) -> str:
        if path not in self.files:
            raise RuntimeError("missing")
        return self.files[path]

    def write_file(self, cid: str, path: str, content: str) -> None:
        self.files[path] = content


@pytest.fixture
def executor() -> SandboxToolExecutor:
    return SandboxToolExecutor(FakeDocker(), "cid", groups=[], caps=Caps())  # type: ignore[arg-type]


def test_tool_specs_have_names() -> None:
    assert {t.name for t in TOOL_SPECS} == {
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "run_tests",
    }


async def test_read_file_confined_to_app(executor: SandboxToolExecutor) -> None:
    res = await executor("read_file", {"path": "../etc/passwd"})
    assert res.is_error and "outside /app" in res.content
    res = await executor("read_file", {"path": "web/a.ts"})
    assert res.content.startswith("1\tconst x")


async def test_edit_file_requires_unique_match(executor: SandboxToolExecutor) -> None:
    res = await executor(
        "edit_file", {"path": "web/a.ts", "old_string": "const", "new_string": "let"}
    )
    assert res.is_error and "2 matches" in res.content
    res = await executor(
        "edit_file",
        {"path": "web/a.ts", "old_string": "const x = 1;", "new_string": "const x = 10;"},
    )
    assert not res.is_error
    assert (await executor("read_file", {"path": "web/a.ts"})).content.startswith(
        "1\tconst x = 10;"
    )


async def test_bash_counts_calls_and_truncates(executor: SandboxToolExecutor) -> None:
    await executor("bash", {"command": "echo hi"})
    assert executor.calls == 1
    executor.docker.files["/app/big.txt"] = "x" * 100_000  # type: ignore[attr-defined]
    res = await executor("read_file", {"path": "big.txt"})
    assert "[truncated" in res.content and len(res.content) < 40_000


async def test_unknown_tool_is_error(executor: SandboxToolExecutor) -> None:
    res = await executor("nope", {})
    assert res.is_error
