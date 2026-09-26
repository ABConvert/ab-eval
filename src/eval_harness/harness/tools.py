from __future__ import annotations

import posixpath
from typing import Any

from eval_harness.adapters.base import Caps, ToolResult, ToolSpec
from eval_harness.config import Runner
from eval_harness.harness.sandbox import Docker

ROOT = "/app"
MAX_TOOL_OUTPUT = 30_000

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="bash",
        description=(
            "Run a shell command inside the repository sandbox (cwd /app, no network). "
            "Returns stdout+stderr and the exit code."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ),
    ToolSpec(
        name="read_file",
        description=(
            "Read a file (path relative to the repo root) with line numbers. "
            "Optional 1-based start_line/end_line."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        name="edit_file",
        description=(
            "Replace exactly one occurrence of old_string with new_string in the file. "
            "Fails if the match is absent or ambiguous."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    ),
    ToolSpec(
        name="run_tests",
        description="Run the task's test files and return the summary and output.",
        input_schema={"type": "object", "properties": {}},
    ),
]


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    half = MAX_TOOL_OUTPUT // 2
    return text[:half] + f"\n…[truncated {len(text) - MAX_TOOL_OUTPUT} chars]…\n" + text[-half:]


class SandboxToolExecutor:
    """Executes model tool calls inside the case container; paths are confined to /app."""

    def __init__(
        self,
        docker: Docker,
        cid: str,
        *,
        groups: list[tuple[Runner, list[str]]],
        caps: Caps,
    ) -> None:
        self.docker = docker
        self.cid = cid
        self.groups = groups
        self.caps = caps
        self.calls = 0
        self.log: list[dict[str, Any]] = []
        self.identity: dict[str, Any] | None = None  # set by the runner for stdio tool servers

    def _abs(self, path: str) -> str:
        full = posixpath.normpath(posixpath.join(ROOT, path))
        if full != ROOT and not full.startswith(ROOT + "/"):
            raise ValueError(f"path {path!r} is outside /app")
        return full

    async def __call__(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls += 1
        try:
            result = await self._dispatch(name, args)
        except Exception as e:
            result = ToolResult(content=f"Error: {e}", is_error=True)
        self.log.append(
            {
                "tool": name,
                "args": args,
                "is_error": result.is_error,
                "output_chars": len(result.content),
            }
        )
        return result

    async def _dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "bash":
            res = self.docker.exec(
                self.cid, str(args["command"]), timeout=self.caps.tool_timeout_seconds
            )
            body = res.stdout + ("\n" + res.stderr if res.stderr else "")
            return ToolResult(content=_truncate(f"{body}\n[exit {res.code}]"), is_error=not res.ok)
        if name == "read_file":
            text = self.docker.read_file(self.cid, self._abs(str(args["path"])))
            lines = text.splitlines()
            start = max(int(args.get("start_line") or 1), 1)
            end = min(int(args.get("end_line") or len(lines)), len(lines))
            numbered = "\n".join(f"{i}\t{lines[i - 1]}" for i in range(start, end + 1))
            return ToolResult(content=_truncate(numbered))
        if name == "write_file":
            self.docker.write_file(self.cid, self._abs(str(args["path"])), str(args["content"]))
            return ToolResult(content=f"wrote {args['path']}")
        if name == "edit_file":
            path = self._abs(str(args["path"]))
            text = self.docker.read_file(self.cid, path)
            old, new = str(args["old_string"]), str(args["new_string"])
            n = text.count(old)
            if n != 1:
                return ToolResult(
                    content=(
                        f"Error: old_string has {n} matches in {args['path']}; "
                        "it must match exactly once"
                    ),
                    is_error=True,
                )
            self.docker.write_file(self.cid, path, text.replace(old, new, 1))
            return ToolResult(content=f"edited {args['path']}")
        if name == "run_tests":
            if not self.groups:
                return ToolResult(content="Error: no test runner configured", is_error=True)
            from eval_harness.harness.testrun import run_groups

            tr = run_groups(
                self.docker, self.cid, self.groups, timeout=self.caps.tool_timeout_seconds
            )
            head = (
                f"tests: {tr.passed}/{tr.total} passed, {tr.failed} failed, "
                f"{tr.skipped} skipped (exit {tr.exit_code})\n"
            )
            return ToolResult(content=_truncate(head + tr.output), is_error=not tr.ok)
        return ToolResult(content=f"Error: unknown tool {name!r}", is_error=True)
