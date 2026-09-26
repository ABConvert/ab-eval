from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from eval_harness.adapters.anthropic import AnthropicAdapter
from eval_harness.adapters.base import Caps, ToolResult, ToolSpec
from eval_harness.adapters.tracing import Tracer
from eval_harness.config import ModelConfig, Prices


def _msg(content: list[Any], stop_reason: str, in_tok: int = 100, out_tok: int = 10) -> Any:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        _request_id="req_test",
    )


def _block(**kw: Any) -> Any:
    ns = SimpleNamespace(**kw)
    ns.model_dump = lambda: dict(kw)
    return ns


class FakeStream:
    def __init__(self, message: Any) -> None:
        self._m = message

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *a: Any) -> None:
        return None

    async def get_final_message(self) -> Any:
        return self._m


class FakeMessages:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        return FakeStream(self.replies.pop(0))


class FakeClient:
    def __init__(self, replies: list[Any]) -> None:
        self.messages = FakeMessages(replies)


async def test_loop_executes_tools_then_stops() -> None:
    replies = [
        _msg([_block(type="tool_use", id="t1", name="bash", input={"command": "ls"})], "tool_use"),
        _msg([_block(type="text", text="done")], "end_turn", in_tok=200, out_tok=20),
    ]
    cfg = ModelConfig(
        key="m",
        provider="anthropic",
        model="claude-opus-5",
        effort="high",
        price_per_mtok=Prices(input=5, output=25, cache_read=0.5, cache_write=6.25),
    )
    client = FakeClient(replies)
    adapter = AnthropicAdapter(cfg, client=client)  # type: ignore[arg-type]
    seen: list[tuple[str, dict[str, Any]]] = []

    async def execute(name: str, args: dict[str, Any]) -> ToolResult:
        seen.append((name, args))
        return ToolResult(content="a.ts")

    res = await adapter.run_agent(
        system="s",
        task="t",
        tools=[ToolSpec(name="bash", description="d", input_schema={"type": "object"})],
        execute=execute,
        caps=Caps(),
        trace=Tracer.noop(),
    )
    assert seen == [("bash", {"command": "ls"})]
    assert res.turns == 2 and res.tool_calls == 1
    assert res.stop_reason == "end_turn" and res.cap_hit is None
    assert res.usage.input_tokens == 300 and res.usage.output_tokens == 30
    assert abs(res.cost_usd - (300 * 5 + 30 * 25) / 1_000_000) < 1e-9
    first = client.messages.requests[0]
    assert first["output_config"] == {"effort": "high"} and "temperature" not in first
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["tools"][-1]["cache_control"] == {"type": "ephemeral"}


async def test_max_turns_cap() -> None:
    replies = [
        _msg(
            [_block(type="tool_use", id=f"t{i}", name="bash", input={"command": "ls"})], "tool_use"
        )
        for i in range(3)
    ]
    cfg = ModelConfig(key="m", provider="anthropic", model="claude-opus-5")
    adapter = AnthropicAdapter(cfg, client=FakeClient(replies))  # type: ignore[arg-type]

    async def execute(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content="ok")

    res = await adapter.run_agent(
        system="s", task="t", tools=[], execute=execute, caps=Caps(max_turns=2), trace=Tracer.noop()
    )
    assert res.cap_hit == "max_turns" and res.turns == 2
