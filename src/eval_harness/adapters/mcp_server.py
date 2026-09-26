"""Stdio MCP server exposing the sandboxed tools to agents that cannot host tools in-process.

Started by the codex-cli adapter as a subprocess:
    python -m eval_harness.adapters.mcp_server --cid <container> --repo <key>
        --groups '[["unit", ["web/a.test.ts"]]]' --log <jsonl path> [--tool-timeout 300]
Every call is appended to the log so the adapter can rebuild the tool log afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from eval_harness.adapters.base import Caps
from eval_harness.config import load_repos
from eval_harness.harness.sandbox import Docker
from eval_harness.harness.tools import TOOL_SPECS, SandboxToolExecutor


def build_server(executor: SandboxToolExecutor, log_path: Path) -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "sandbox", instructions="Tools that act inside the task's repository sandbox."
    )
    specs = {t.name: t for t in TOOL_SPECS}

    async def call(name: str, args: dict[str, Any]) -> str:
        res = await executor(name, args)
        with log_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "tool": name,
                        "args": args,
                        "is_error": res.is_error,
                        "output_chars": len(res.content),
                    }
                )
                + "\n"
            )
        if res.is_error:
            raise RuntimeError(res.content)
        return res.content

    @server.tool(name="bash", description=specs["bash"].description)
    async def bash(command: str) -> str:
        return await call("bash", {"command": command})

    @server.tool(name="read_file", description=specs["read_file"].description)
    async def read_file(
        path: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        args: dict[str, Any] = {"path": path}
        if start_line is not None:
            args["start_line"] = start_line
        if end_line is not None:
            args["end_line"] = end_line
        return await call("read_file", args)

    @server.tool(name="write_file", description=specs["write_file"].description)
    async def write_file(path: str, content: str) -> str:
        return await call("write_file", {"path": path, "content": content})

    @server.tool(name="edit_file", description=specs["edit_file"].description)
    async def edit_file(path: str, old_string: str, new_string: str) -> str:
        return await call(
            "edit_file", {"path": path, "old_string": old_string, "new_string": new_string}
        )

    @server.tool(name="run_tests", description=specs["run_tests"].description)
    async def run_tests() -> str:
        return await call("run_tests", {})

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--groups", required=True, help="JSON [[runner, [files...]], ...]")
    parser.add_argument("--log", required=True)
    parser.add_argument("--tool-timeout", type=int, default=300)
    ns = parser.parse_args(argv)
    repo = load_repos()[ns.repo]
    groups = [(repo.runners[name], list(files)) for name, files in json.loads(ns.groups)]
    executor = SandboxToolExecutor(
        Docker(), ns.cid, groups=groups, caps=Caps(tool_timeout_seconds=ns.tool_timeout)
    )
    build_server(executor, Path(ns.log)).run("stdio")


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    main()
