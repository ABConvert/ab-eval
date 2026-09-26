"""What the openai-compat adapter actually puts on the wire, and what it records.

The adapter used to build `params = {"effort": ...}` for the trace and then post a body
containing only model, messages, max_tokens and tools. Two bench-v2 runs were configured
`effort: high`, ran at their server's default, and recorded the effort as though it had been
honoured. These tests are about the gap between the two: nothing is asserted about config
that is not also asserted about the request.

No network: every call is answered by a fake transport.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from eval_harness.adapters.base import Caps, ToolResult
from eval_harness.adapters.openai_compat import OpenAICompatAdapter, reasoning_body
from eval_harness.adapters.tracing import Tracer
from eval_harness.config import ModelConfig

REPLY = {
    "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
}


class FakeTransport(httpx.AsyncBaseTransport):
    """Answers every POST from a script, and keeps the bodies that were sent."""

    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status = status
        self.body = REPLY if body is None else body
        self.sent: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json

        self.sent.append(json.loads(request.content))
        return httpx.Response(self.status, json=self.body)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_API_KEY", "not-a-real-key")


def _cfg(**over: Any) -> ModelConfig:
    base: dict[str, Any] = {
        "key": "demo-compat",
        "provider": "openai-compat",
        "model": "demo-1",
        "base_url": "https://example.invalid/v1",
        "api_key_env": "DEMO_API_KEY",
        "effort": "high",
    }
    return ModelConfig(**{**base, **over})


async def _run(cfg: ModelConfig, transport: FakeTransport) -> Any:
    adapter = OpenAICompatAdapter(cfg)

    async def execute(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content="ok")

    real = httpx.AsyncClient

    def patched(*a: Any, **kw: Any) -> httpx.AsyncClient:
        return real(*a, **{**kw, "transport": transport})

    import eval_harness.adapters.openai_compat as mod

    original = mod.httpx.AsyncClient
    mod.httpx.AsyncClient = patched  # type: ignore[misc]
    try:
        return await adapter.run_agent(
            system="s", task="t", tools=[], execute=execute, caps=Caps(), trace=Tracer.noop()
        )
    finally:
        mod.httpx.AsyncClient = original  # type: ignore[misc]


# ------------------------------------------------------------------- the reasoning field


def test_the_body_fragment_for_each_shape() -> None:
    assert reasoning_body("omit", "high") == {}
    assert reasoning_body("reasoning_effort", "high") == {"reasoning_effort": "high"}
    assert reasoning_body("reasoning.effort", "high") == {"reasoning": {"effort": "high"}}


async def test_no_reasoning_field_is_sent_by_default() -> None:
    """The default has to be what a plain llama.cpp or ollama shim already accepted."""
    t = FakeTransport()
    result = await _run(_cfg(), t)
    body = t.sent[0]
    assert "reasoning_effort" not in body
    assert "reasoning" not in body
    # And the record says so, rather than repeating the configured effort as if it landed.
    assert result.request_settings["reasoning_param"] == "omit"
    assert result.request_settings["reasoning_sent"] is None


async def test_reasoning_effort_is_sent_when_the_model_declares_it() -> None:
    t = FakeTransport()
    result = await _run(_cfg(reasoning_param="reasoning_effort", effort="medium"), t)
    assert t.sent[0]["reasoning_effort"] == "medium"
    assert result.request_settings["reasoning_sent"] == {"reasoning_effort": "medium"}


async def test_the_reasoning_object_shape_is_sent_when_declared() -> None:
    t = FakeTransport()
    result = await _run(_cfg(reasoning_param="reasoning.effort"), t)
    assert t.sent[0]["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in t.sent[0]
    assert result.request_settings["reasoning_sent"] == {"reasoning": {"effort": "high"}}


async def test_a_rejected_reasoning_field_fails_legibly() -> None:
    """A raw 400 dump does not tell you the harness added a field your server refuses."""
    t = FakeTransport(status=400, body={"error": {"message": "unknown field reasoning_effort"}})
    with pytest.raises(RuntimeError) as caught:
        await _run(_cfg(reasoning_param="reasoning_effort"), t)
    message = str(caught.value)
    assert "reasoning_effort" in message  # the field
    assert "https://example.invalid/v1/chat/completions" in message  # the endpoint
    assert "reasoning_param: omit" in message  # and how to stop sending it
    assert "not-a-real-key" not in message


async def test_a_rejection_with_no_reasoning_field_keeps_the_plain_message() -> None:
    """Nothing was added, so nothing is claimed about a field; the server speaks for itself."""
    t = FakeTransport(status=500, body={"error": "upstream exploded"})
    with pytest.raises(RuntimeError) as caught:
        await _run(_cfg(), t)
    message = str(caught.value)
    assert "reasoning_param" not in message
    assert "upstream exploded" in message


# ------------------------------------------------------------------------- temperature


async def test_no_temperature_is_sent_when_it_is_unset() -> None:
    """Unset must behave exactly as before this field existed: the key is simply absent."""
    t = FakeTransport()
    result = await _run(_cfg(), t)
    assert "temperature" not in t.sent[0]
    assert result.request_settings["temperature"] is None


async def test_temperature_is_sent_when_it_is_set() -> None:
    t = FakeTransport()
    result = await _run(_cfg(temperature=0.2), t)
    assert t.sent[0]["temperature"] == 0.2
    assert result.request_settings["temperature"] == 0.2


async def test_temperature_zero_is_sent_rather_than_treated_as_unset() -> None:
    """0.0 is a real, deliberate setting; `if temperature:` would drop it."""
    t = FakeTransport()
    await _run(_cfg(temperature=0.0), t)
    assert t.sent[0]["temperature"] == 0.0


# ------------------------------------------------ a config with none of the new fields


async def test_a_config_with_no_new_fields_sends_exactly_what_it_used_to() -> None:
    """The backward-compatibility assertion, written as the whole request body."""
    t = FakeTransport()
    await _run(_cfg(), t)
    assert set(t.sent[0]) == {"model", "messages", "max_tokens"}
    assert t.sent[0]["max_tokens"] == 16000


# ------------------------------------------------ fields the other providers cannot send


@pytest.mark.parametrize("provider", ["anthropic", "claude-code", "codex-cli"])
@pytest.mark.parametrize(
    "field,value", [("temperature", 0.2), ("reasoning_param", "reasoning_effort")]
)
def test_a_field_only_openai_compat_can_send_is_refused_elsewhere(
    provider: str, field: str, value: Any
) -> None:
    """Better an error at load than a value snapshotted into a run that never sent it."""
    with pytest.raises(ValueError, match=field):
        ModelConfig(key="k", provider=provider, model="m", **{field: value})


@pytest.mark.parametrize("provider", ["anthropic", "claude-code", "codex-cli", "openai-compat"])
def test_leaving_them_unset_is_fine_on_every_provider(provider: str) -> None:
    cfg = ModelConfig(key="k", provider=provider, model="m")
    assert cfg.temperature is None and cfg.reasoning_param == "omit"
