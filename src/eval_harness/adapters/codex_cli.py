from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval_harness import PROJECT_ROOT
from eval_harness.adapters.base import (
    NOT_A_PARAMETER,
    AgentResult,
    Caps,
    ToolExecutor,
    ToolSpec,
    Usage,
)
from eval_harness.adapters.pricing import cost_usd
from eval_harness.adapters.tracing import Tracer
from eval_harness.config import ModelConfig

# Codex features switched off. code_mode_host must stay on: in this Codex build every tool
# call, including MCP calls, is routed through code mode and fails closed without it. Its JS
# executor therefore remains reachable; the sandbox is workspace-write on an empty scratch
# directory with no network, and every non-MCP execution item is recorded as a host_exec item.
DISABLED_FEATURES = [
    "shell_tool",
    "unified_exec",
    "apps",
    "browser_use",
    "image_generation",
    "computer_use",
    "collaboration_modes",
]
HOST_EXEC_ITEM_TYPES = {"command_execution", "code_execution", "exec"}
EFFORTS = {"low", "medium", "high", "xhigh", "minimal"}


def codex_version() -> str:
    proc = subprocess.run(["codex", "--version"], capture_output=True, text=True)
    return proc.stdout.strip() or "unknown"


def usage_from_events(usage: dict[str, Any] | None) -> Usage:
    """Codex reports input_tokens inclusive of cached tokens; store the uncached remainder."""
    u = usage or {}
    cached = int(u.get("cached_input_tokens") or 0)
    return Usage(
        input_tokens=max(int(u.get("input_tokens") or 0) - cached, 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cache_read_tokens=cached,
        cache_write_tokens=int(u.get("cache_write_input_tokens") or 0),
    )


def parse_events(lines: list[str]) -> dict[str, Any]:
    """Fold `codex exec --json` JSONL into usage, tool calls, final text and failure info."""
    usage: dict[str, Any] = {}
    tool_calls = 0
    turns = 0
    final_text = ""
    failed: str | None = None
    host_exec: list[dict[str, Any]] = []
    other_item_types: dict[str, int] = {}
    for raw in lines:
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        item = ev.get("item") or {}
        if kind == "item.completed":
            itype = str(item.get("type"))
            if itype == "mcp_tool_call":
                tool_calls += 1
            elif itype == "agent_message":
                turns += 1
                final_text = str(item.get("text") or final_text)
            elif itype in HOST_EXEC_ITEM_TYPES:
                host_exec.append({k: str(v)[:500] for k, v in item.items() if k != "id"})
            elif itype != "reasoning":
                other_item_types[itype] = other_item_types.get(itype, 0) + 1
        elif kind == "turn.completed":
            for k, v in (ev.get("usage") or {}).items():
                usage[k] = usage.get(k, 0) + int(v or 0)
        elif kind in ("turn.failed", "error"):
            failed = json.dumps({k: v for k, v in ev.items() if k != "type"})[:2000]
    return {
        "usage": usage,
        "tool_calls": tool_calls,
        "turns": turns,
        "final_text": final_text,
        "failed": failed,
        "host_exec": host_exec,
        "other_item_types": other_item_types,
    }


def read_tool_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# What the sandbox MCP server needs from the harness's environment. Codex starts MCP
# servers with a minimal environment plus `mcp_servers.<name>.env`, so anything else is
# gone: without ABEVAL_DATA_ROOT the server read the checkout's (absent) repos.yaml, died
# before serving a tool, and every attempt ran with no sandbox tools at all. None of these
# hold a credential; provider keys stay out on purpose.
MCP_ENV_PREFIXES = ("ABEVAL_",)
MCP_ENV_NAMES = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")


def mcp_env_table(environ: Mapping[str, str]) -> str:
    """The MCP server's environment as a TOML inline table (Codex rejects JSON objects)."""
    keep = {
        k: v
        for k, v in sorted(environ.items())
        if k in MCP_ENV_NAMES or k.startswith(MCP_ENV_PREFIXES)
    }
    keep.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return "{ " + ", ".join(f"{k} = {json.dumps(v)}" for k, v in keep.items()) + " }"


class CodexCliAdapter:
    """Runs a model through `codex exec` on the local ChatGPT login.

    Codex's own execution surfaces (shell, unified exec, code mode, apps) are disabled by
    feature flag, web search and hooks are off, the working root is an empty scratch
    directory under the read-only sandbox, and the only tools are the harness's sandbox
    tools served over stdio by `eval_harness.adapters.mcp_server`, which executes inside
    the case container. The harness system prompt replaces Codex's built-in instructions.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.name = cfg.key

    def command(
        self,
        *,
        scratch: Path,
        instructions: Path,
        mcp_args: list[str],
        log_path: Path,
        tool_timeout: int,
    ) -> list[str]:
        effort = self.cfg.effort if self.cfg.effort in EFFORTS else "high"
        server_args = json.dumps(["-m", "eval_harness.adapters.mcp_server", *mcp_args])
        env_table = mcp_env_table(os.environ)
        cmd = [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "-C",
            str(scratch),
            "--approve-for-me",  # workspace-write sandbox + automatic approval review
            "-m",
            self.cfg.model,
        ]
        for feature in DISABLED_FEATURES:
            cmd += ["--disable", feature]
        overrides = [
            f"model_reasoning_effort={json.dumps(effort)}",
            'web_search="disabled"',  # top-level key; tools.web_search does not disable it
            "tools.view_image=false",
            "features.hooks=false",
            "project_doc_max_bytes=0",
            f"model_instructions_file={json.dumps(str(instructions))}",
            f"mcp_servers.sandbox.command={json.dumps(sys.executable)}",
            f"mcp_servers.sandbox.args={server_args}",
            f"mcp_servers.sandbox.cwd={json.dumps(str(PROJECT_ROOT))}",
            f"mcp_servers.sandbox.env={env_table}",
            "mcp_servers.sandbox.startup_timeout_sec=60",
            f"mcp_servers.sandbox.tool_timeout_sec={tool_timeout + 30}",
        ]
        for o in overrides:
            cmd += ["-c", o]
        cmd.append("-")  # prompt on stdin
        return cmd

    async def run_agent(
        self,
        *,
        system: str,
        task: str,
        tools: list[ToolSpec],
        execute: ToolExecutor,
        caps: Caps,
        trace: Tracer,
    ) -> AgentResult:
        # The executor is reached through the stdio MCP server; the harness passes the
        # container identity on `execute` (see runner.sandbox_identity).
        identity = getattr(execute, "identity", None)
        if identity is None:
            raise RuntimeError("codex-cli adapter needs a SandboxToolExecutor with identity")
        scratch = Path(tempfile.mkdtemp(prefix="abeval-codex-"))
        instructions = scratch / "instructions.md"
        instructions.write_text(system)
        log_path = scratch / "tool-log.jsonl"
        mcp_args = [
            "--cid",
            identity["cid"],
            "--repo",
            identity["repo"],
            "--groups",
            json.dumps(identity["groups"]),
            "--log",
            str(log_path),
            "--tool-timeout",
            str(caps.tool_timeout_seconds),
        ]
        cmd = self.command(
            scratch=scratch / "work",
            instructions=instructions,
            mcp_args=mcp_args,
            log_path=log_path,
            tool_timeout=caps.tool_timeout_seconds,
        )
        (scratch / "work").mkdir()
        version = codex_version()
        settings: dict[str, Any] = {
            "provider": "codex-cli",
            "harness": version,
            "model": self.cfg.model,
            "effort": self.cfg.effort,
            # `codex exec` has no override for either: its only max_output_tokens keys are
            # budgets for tool output, and it exposes no temperature at all (checked against
            # codex-cli 0.155.1). config.ModelConfig refuses a temperature on this provider;
            # max_output_tokens it cannot refuse, because every entry carries one.
            "max_output_tokens": NOT_A_PARAMETER,
            "temperature": NOT_A_PARAMETER,
            "builtin_tools": "disabled: " + ", ".join(DISABLED_FEATURES),
            "sandbox": "workspace-write on an empty scratch root, no network (--approve-for-me)",
        }
        started = time.monotonic()
        cap_hit: str | None = None
        params = {"effort": self.cfg.effort}
        with trace.generation("codex-cli", model=self.cfg.model, model_parameters=params) as gen:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(scratch / "work"),
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(task.encode()), timeout=caps.wall_clock_seconds
                )
            except TimeoutError:
                proc.kill()
                out, err = await proc.communicate()
                cap_hit = "wall_clock"
            parsed = parse_events(out.decode(errors="replace").splitlines())
            usage = usage_from_events(parsed["usage"])
            priced = cost_usd(usage, self.cfg.price_per_mtok)
            gen.end(
                input={"task_chars": len(task)},
                output={"turns": parsed["turns"], "failed": parsed["failed"]},
                usage=usage,
                cost=priced,
            )
        tool_log = read_tool_log(log_path)
        stop_reason = "wall_clock" if cap_hit else ("error" if parsed["failed"] else "end_turn")
        if proc.returncode not in (0, None) and not cap_hit:
            stop_reason = "error"
            if parsed["turns"] == 0 and not tool_log:
                tail = err.decode(errors="replace")[-1500:]
                raise RuntimeError(f"codex exec exited {proc.returncode} before any turn: {tail}")
        settings["exit_code"] = proc.returncode
        settings["stderr_tail"] = err.decode(errors="replace")[-2000:]
        settings["failed_event"] = parsed["failed"]
        settings["host_exec_items"] = parsed["host_exec"]
        settings["other_item_types"] = parsed["other_item_types"]
        settings["wall_clock_seconds"] = round(time.monotonic() - started, 1)
        settings["tool_log"] = tool_log
        return AgentResult(
            turns=parsed["turns"],
            tool_calls=len(tool_log) or parsed["tool_calls"],
            usage=usage,
            cost_usd=priced,
            stop_reason=stop_reason,
            cap_hit=cap_hit,
            final_text=parsed["final_text"],
            transcript=[],
            request_settings=settings,
        )
