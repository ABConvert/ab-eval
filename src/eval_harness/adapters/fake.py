from __future__ import annotations

from typing import Any

from eval_harness.adapters.base import AgentResult, Caps, ToolExecutor, ToolSpec, Usage


class NoopAdapter:
    """Does nothing. Proves a case fails without a fix."""

    name = "noop"

    async def run_agent(
        self,
        *,
        system: str,
        task: str,
        tools: list[ToolSpec],
        execute: ToolExecutor,
        caps: Caps,
        trace: Any,
    ) -> AgentResult:
        return AgentResult(
            turns=0,
            tool_calls=0,
            usage=Usage(),
            cost_usd=0.0,
            stop_reason="end_turn",
            cap_hit=None,
            final_text="",
            request_settings={"adapter": "noop"},
        )


class HumanPatchAdapter:
    """Applies the reference patch through the same tools a model would use. Spends nothing."""

    name = "human-patch"

    def __init__(self, patch: str | dict[str, str]) -> None:
        self.patches = patch if isinstance(patch, dict) else {"*": patch}

    async def run_agent(
        self,
        *,
        system: str,
        task: str,
        tools: list[ToolSpec],
        execute: ToolExecutor,
        caps: Caps,
        trace: Any,
    ) -> AgentResult:
        identity = getattr(execute, "identity", None) or {}
        case_id = str(identity.get("case_id") or "*")
        patch = (
            self.patches.get(case_id) or self.patches.get("*") or next(iter(self.patches.values()))
        )
        await execute("write_file", {"path": ".abeval-human.patch", "content": patch})
        res = await execute(
            "bash", {"command": "git apply .abeval-human.patch && rm .abeval-human.patch"}
        )
        return AgentResult(
            turns=1,
            tool_calls=2,
            usage=Usage(),
            cost_usd=0.0,
            stop_reason="end_turn" if not res.is_error else "error",
            cap_hit=None,
            final_text=res.content,
            request_settings={"adapter": "human-patch"},
        )
