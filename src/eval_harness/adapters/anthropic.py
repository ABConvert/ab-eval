from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from anthropic import AsyncAnthropic

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


def _usage(u: Any) -> Usage:
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
    )


def _dump_blocks(content: list[Any]) -> list[dict[str, Any]]:
    return [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in content]


class AnthropicAdapter:
    """Manual agent loop on the Messages API. No fallbacks, no temperature, adaptive thinking."""

    def __init__(self, cfg: ModelConfig, client: AsyncAnthropic | None = None) -> None:
        self.cfg = cfg
        self.name = cfg.key
        self.client = client or AsyncAnthropic(max_retries=4)

    def _tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        if out:
            out[-1]["cache_control"] = {"type": "ephemeral"}
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
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        usage = Usage()
        turns = tool_calls = 0
        cap_hit: str | None = None
        stop_reason = "none"
        final_text = ""
        started = time.monotonic()
        settings = {
            "model": self.cfg.model,
            "effort": self.cfg.effort,
            "max_tokens": self.cfg.max_output_tokens,
            "thinking": "adaptive (default)",
            # anthropic.resources.messages.create takes no temperature; config.ModelConfig
            # refuses one on this provider rather than record a value nothing sends.
            "temperature": NOT_A_PARAMETER,
        }
        while True:
            if turns >= caps.max_turns:
                cap_hit = "max_turns"
                break
            if usage.output_tokens >= caps.max_output_tokens_total:
                cap_hit = "max_output_tokens_total"
                break
            if time.monotonic() - started >= caps.wall_clock_seconds:
                cap_hit = "wall_clock"
                break
            turns += 1
            request: dict[str, Any] = {
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_output_tokens,
                "system": [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ],
                "messages": messages,
                "tools": self._tools(tools),
                "output_config": {"effort": self.cfg.effort},
            }
            remaining = caps.wall_clock_seconds - (time.monotonic() - started)
            params = {"effort": self.cfg.effort, "max_tokens": self.cfg.max_output_tokens}
            with trace.generation(
                f"turn-{turns}", model=self.cfg.model, model_parameters=params
            ) as gen:
                try:
                    async with self.client.messages.stream(**request) as stream:
                        message = await asyncio.wait_for(
                            stream.get_final_message(), timeout=max(remaining, 30)
                        )
                except TimeoutError:
                    cap_hit = "wall_clock"
                    gen.end(input=None, output="timeout", usage=Usage(), cost=0.0)
                    break
                turn_usage = _usage(message.usage)
                usage.add(turn_usage)
                gen.end(
                    input={"turn": turns, "messages": len(messages)},
                    output=_dump_blocks(message.content),
                    usage=turn_usage,
                    cost=cost_usd(turn_usage, self.cfg.price_per_mtok),
                )
            stop_reason = str(message.stop_reason)
            blocks: list[Any] = list(message.content)
            messages.append({"role": "assistant", "content": _dump_blocks(blocks)})
            texts = [b.text for b in blocks if getattr(b, "type", "") == "text"]
            if texts:
                final_text = "\n".join(texts)
            if stop_reason == "pause_turn":
                continue
            if stop_reason != "tool_use":
                break
            uses = [b for b in blocks if getattr(b, "type", "") == "tool_use"]
            results: list[dict[str, Any]] = []
            for use in uses:
                tool_calls += 1
                args = use.input if isinstance(use.input, dict) else json.loads(use.input)
                with trace.tool_span(use.name, args):
                    res = await execute(use.name, args)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use.id,
                        "content": res.content,
                        "is_error": res.is_error,
                    }
                )
            messages.append({"role": "user", "content": results})
        settings["wall_clock_seconds"] = round(time.monotonic() - started, 1)
        return AgentResult(
            turns=turns,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=cost_usd(usage, self.cfg.price_per_mtok),
            stop_reason=stop_reason,
            cap_hit=cap_hit,
            final_text=final_text,
            transcript=messages,
            request_settings=settings,
        )
