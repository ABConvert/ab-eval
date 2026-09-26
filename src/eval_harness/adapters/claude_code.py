from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from typing import Any

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

SERVER = "sandbox"

# The Agent SDK has no max-output-tokens option and the `claude` CLI has no flag for it, but
# the CLI reads this variable and `ClaudeAgentOptions.env` is merged into the environment it
# is spawned with. Checked against claude-agent-sdk 0.2.152 and codex-cli 0.155.1, not
# assumed. The CLI clamps the value to the model's own ceiling rather than failing.
MAX_OUTPUT_TOKENS_ENV = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"


def claude_code_version() -> str:
    proc = subprocess.run(["claude", "--version"], capture_output=True, text=True)
    return proc.stdout.strip() or "unknown"


def usage_from_result(usage: dict[str, Any] | None) -> Usage:
    u = usage or {}
    return Usage(
        input_tokens=int(u.get("input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cache_read_tokens=int(u.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
    )


class RateLimitedError(RuntimeError):
    """The subscription rejected the request; `resets_at` is a Unix timestamp."""

    def __init__(self, message: str, resets_at: int | None) -> None:
        super().__init__(message)
        self.resets_at = resets_at


def rate_limit_rejection(events: list[dict[str, Any]]) -> int | None:
    """Reset time of the first rate-limit event whose status is `rejected`, else None."""
    for ev in events:
        info = ev.get("rate_limit_info") or {}
        if isinstance(info, dict) and info.get("status") == "rejected":
            return int(info.get("resetsAt") or info.get("resets_at") or 0)
    return None


def cap_from_subtype(subtype: str | None) -> str | None:
    if subtype == "error_max_turns":
        return "max_turns"
    if subtype == "error_max_budget_usd":
        return "max_budget"
    return None


def mcp_tool_names(tools: list[ToolSpec]) -> list[str]:
    return [f"mcp__{SERVER}__{t.name}" for t in tools]


def request_settings(cfg: ModelConfig, caps: Caps, harness: str) -> dict[str, Any]:
    """What this provider was asked for, as the attempt record will carry it.

    Its own function so that what the record claims is one thing a test can read, rather
    than a dict built inside a coroutine that has to reach the network to return it. Every
    declared field appears here either as the value in force or as `NOT_A_PARAMETER` — the
    point being that a reader can never mistake "we did not send this" for "we sent the
    default", which is how a configured effort came to look honoured for two whole runs.
    """
    return {
        "provider": "claude-code",
        "harness": harness,
        "model": cfg.model,
        "effort": cfg.effort,
        "max_turns": caps.max_turns,
        "max_output_tokens": cfg.max_output_tokens,
        "max_output_tokens_via": (
            f"{MAX_OUTPUT_TOKENS_ENV} on the CLI the SDK spawns; the CLI clamps it to the "
            "model's own ceiling"
        ),
        "temperature": NOT_A_PARAMETER,
        "thinking": "claude-code default",
        "builtin_tools": "disabled",
    }


class ClaudeCodeAdapter:
    """Runs the model through the Claude Agent SDK on the local Claude Code login.

    Built-in tools are disabled (`tools=[]`) and only the harness's sandboxed tools are
    exposed through an in-process MCP server, so generated code still runs only inside
    the case container. User settings, CLAUDE.md files and hooks are not loaded.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.name = cfg.key

    def _sdk_tools(self, tools: list[ToolSpec], execute: ToolExecutor) -> list[Any]:
        from claude_agent_sdk import tool

        out: list[Any] = []
        for spec in tools:

            async def handler(args: dict[str, Any], _name: str = spec.name) -> dict[str, Any]:
                res = await execute(_name, args)
                return {
                    "content": [{"type": "text", "text": res.content}],
                    "is_error": res.is_error,
                }

            out.append(tool(spec.name, spec.description, spec.input_schema)(handler))
        return out

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
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            RateLimitEvent,
            ResultMessage,
            create_sdk_mcp_server,
        )

        # The SDK spawns the `claude` CLI, which refuses to nest inside an interactive session.
        os.environ.pop("CLAUDECODE", None)
        cc_version = claude_code_version()
        workdir = tempfile.mkdtemp(prefix="abeval-cc-")
        options = ClaudeAgentOptions(
            model=self.cfg.model,
            effort=self.cfg.effort,  # type: ignore[arg-type]
            max_turns=caps.max_turns,
            system_prompt=system,
            setting_sources=[],
            tools=[],
            allowed_tools=mcp_tool_names(tools),
            permission_mode="dontAsk",  # unlisted tools are denied, never prompted
            strict_mcp_config=True,  # ignore MCP servers from ~/.claude.json
            mcp_servers={
                SERVER: create_sdk_mcp_server(SERVER, tools=self._sdk_tools(tools, execute))
            },
            cwd=workdir,
            env={MAX_OUTPUT_TOKENS_ENV: str(self.cfg.max_output_tokens)},
        )
        settings: dict[str, Any] = request_settings(self.cfg, caps, cc_version)
        transcript: list[dict[str, Any]] = []
        rate_limits: list[dict[str, Any]] = []
        rejections: list[Any] = []
        result_error: str | None = None
        tool_calls = 0
        turns = 0
        final_text = ""
        stop_reason = "none"
        cap_hit: str | None = None
        usage = Usage()
        reported_cost: float | None = None
        started = time.monotonic()
        params = {"effort": self.cfg.effort, "max_turns": caps.max_turns}
        with trace.generation("claude-code", model=self.cfg.model, model_parameters=params) as gen:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(task)

                async def consume() -> None:
                    nonlocal tool_calls, turns, final_text, stop_reason, usage, reported_cost
                    nonlocal result_error
                    async for msg in client.receive_response():
                        if isinstance(msg, AssistantMessage):
                            turns += 1
                            blocks: list[dict[str, Any]] = []
                            for block in msg.content:
                                data = {"type": type(block).__name__, **vars(block)}
                                blocks.append(data)
                                if type(block).__name__ == "ToolUseBlock":
                                    tool_calls += 1
                                elif type(block).__name__ == "TextBlock":
                                    final_text = str(getattr(block, "text", ""))
                            transcript.append({"role": "assistant", "content": blocks})
                        elif isinstance(msg, RateLimitEvent):
                            rate_limits.append({k: str(v) for k, v in vars(msg).items()})
                            info = msg.rate_limit_info
                            if getattr(info, "status", None) == "rejected":
                                rejections.append(info)
                        elif isinstance(msg, ResultMessage):
                            if msg.is_error and not cap_from_subtype(msg.subtype):
                                result_error = f"{msg.subtype}: {msg.result or msg.errors}"
                            usage = usage_from_result(msg.usage)
                            reported_cost = msg.total_cost_usd
                            turns = int(msg.num_turns or turns)  # SDK turn = one API round trip
                            stop_reason = str(msg.stop_reason or msg.subtype)
                            if msg.result:
                                final_text = msg.result
                            hit = cap_from_subtype(msg.subtype)
                            if hit:
                                nonlocal_cap[0] = hit

                nonlocal_cap: list[str | None] = [None]
                try:
                    await asyncio.wait_for(consume(), timeout=caps.wall_clock_seconds)
                except TimeoutError:
                    cap_hit = "wall_clock"
                    stop_reason = "wall_clock"
                    await client.interrupt()
                cap_hit = cap_hit or nonlocal_cap[0]
            priced = cost_usd(usage, self.cfg.price_per_mtok)
            gen.end(
                input={"task_chars": len(task)},
                output={"stop_reason": stop_reason, "turns": turns},
                usage=usage,
                cost=reported_cost if reported_cost is not None else priced,
            )
        settings["cost_usd_reported"] = reported_cost
        settings["cost_usd_from_prices"] = priced
        settings["wall_clock_seconds"] = round(time.monotonic() - started, 1)
        settings["rate_limit_events"] = rate_limits
        if rejections:
            reset = getattr(rejections[0], "resets_at", None)
            raise RateLimitedError(
                f"rate limited ({getattr(rejections[0], 'rate_limit_type', '?')}); "
                f"resets_at={reset}: {final_text}",
                int(reset) if reset else None,
            )
        if result_error:
            raise RuntimeError(f"claude-code error result: {result_error}")
        return AgentResult(
            turns=turns,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=reported_cost if reported_cost is not None else priced,
            stop_reason=stop_reason,
            cap_hit=cap_hit,
            final_text=final_text,
            transcript=transcript,
            request_settings=settings,
        )
