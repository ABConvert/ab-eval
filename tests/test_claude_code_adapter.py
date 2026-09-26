from typing import Any

from eval_harness.adapters.base import ToolSpec
from eval_harness.adapters.claude_code import (
    cap_from_subtype,
    mcp_tool_names,
    rate_limit_rejection,
    usage_from_result,
)


def test_usage_mapping_from_sdk_result() -> None:
    u = usage_from_result(
        {
            "input_tokens": 457,
            "output_tokens": 4,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 20,
            "server_tool_use": {"web_search_requests": 0},
        }
    )
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (
        457,
        4,
        10,
        20,
    )
    assert usage_from_result(None).input_tokens == 0


def test_cap_and_tool_names() -> None:
    assert cap_from_subtype("error_max_turns") == "max_turns"
    assert cap_from_subtype("success") is None
    specs = [
        ToolSpec(name="bash", description="d", input_schema={}),
        ToolSpec(name="run_tests", description="d", input_schema={}),
    ]
    assert mcp_tool_names(specs) == ["mcp__sandbox__bash", "mcp__sandbox__run_tests"]


def test_rate_limit_rejection_reads_reset_time() -> None:
    events = [
        {"rate_limit_info": {"status": "allowed_warning", "resetsAt": 1}},
        {"rate_limit_info": {"status": "rejected", "resetsAt": 1789035600}},
    ]
    assert rate_limit_rejection(events) == 1789035600
    assert rate_limit_rejection([{"rate_limit_info": {"status": "allowed"}}]) is None
    assert rate_limit_rejection([]) is None


def test_the_sdk_has_no_max_output_tokens_option() -> None:
    """The premise of how the ceiling is applied, asserted rather than remembered.

    `ClaudeAgentOptions` carries no output-token field and the `claude` CLI has no flag for
    one, so the value travels as an environment variable on the process the SDK spawns. If a
    future SDK grows a real option this fails, which is the moment to stop using the variable.
    """
    import dataclasses

    from claude_agent_sdk import ClaudeAgentOptions

    names = {f.name for f in dataclasses.fields(ClaudeAgentOptions)}
    assert "max_output_tokens" not in names
    assert "max_tokens" not in names
    assert "env" in names, "and this is the seam the ceiling actually travels through"


def test_the_declared_ceiling_reaches_the_spawned_cli() -> None:
    """Three bench-v2 runs declared 16000 and nothing carried it anywhere."""
    from eval_harness.adapters.base import Caps
    from eval_harness.adapters.claude_code import (
        MAX_OUTPUT_TOKENS_ENV,
        ClaudeCodeAdapter,
        request_settings,
    )
    from eval_harness.config import ModelConfig

    cfg = ModelConfig(key="k", provider="claude-code", model="m", max_output_tokens=8000)
    options = _options_for(ClaudeCodeAdapter(cfg))
    assert options.env[MAX_OUTPUT_TOKENS_ENV] == "8000"
    # And the attempt record says both the number and how it got there, because 16000 on its
    # own is exactly what the old records claimed while sending nothing at all.
    settings = request_settings(cfg, Caps(), "fake")
    assert settings["max_output_tokens"] == 8000
    assert MAX_OUTPUT_TOKENS_ENV in settings["max_output_tokens_via"]


def test_temperature_is_recorded_as_inapplicable_not_as_unset() -> None:
    """`None` reads as "left at the default"; claude-code has no temperature to leave."""
    from eval_harness.adapters.base import NOT_A_PARAMETER, Caps
    from eval_harness.adapters.claude_code import request_settings
    from eval_harness.config import ModelConfig

    cfg = ModelConfig(key="k", provider="claude-code", model="m")
    assert request_settings(cfg, Caps(), "fake")["temperature"] == NOT_A_PARAMETER


def _options_for(adapter: object) -> Any:
    """The ClaudeAgentOptions the adapter builds, without spawning anything.

    `run_agent` constructs the options and then connects, so the connection is what is
    faked: the client records what it was handed and raises on entry. Nothing spawns, no
    network is touched, and what the test reads is the object the real SDK would receive.
    """
    import asyncio
    import contextlib

    import claude_agent_sdk

    import eval_harness.adapters.claude_code as mod
    from eval_harness.adapters.base import Caps, ToolResult
    from eval_harness.adapters.tracing import Tracer

    captured: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    class FakeClient:
        def __init__(self, options: Any) -> None:
            captured["options"] = options

        async def __aenter__(self) -> "FakeClient":
            raise _Stop

        async def __aexit__(self, *a: Any) -> None:
            return None

    async def execute(name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content="")

    real_client = claude_agent_sdk.ClaudeSDKClient
    real_version = mod.claude_code_version
    claude_agent_sdk.ClaudeSDKClient = FakeClient  # type: ignore[misc,assignment]
    mod.claude_code_version = lambda: "fake"  # type: ignore[assignment]
    try:
        with contextlib.suppress(_Stop):
            asyncio.run(
                adapter.run_agent(  # type: ignore[attr-defined]
                    system="s",
                    task="t",
                    tools=[],
                    execute=execute,
                    caps=Caps(),
                    trace=Tracer.noop(),
                )
            )
    finally:
        claude_agent_sdk.ClaudeSDKClient = real_client  # type: ignore[misc]
        mod.claude_code_version = real_version  # type: ignore[assignment]
    return captured["options"]
