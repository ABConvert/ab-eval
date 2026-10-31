"""Every one-shot branch reports text, usage, cost, latency and what it actually sent.

A prompt task scores one completion per case, so a branch that returned only the words would
leave the leaderboard without cost or speed for that provider. Each provider is faked at its
own seam (the SDK's `query`, the Anthropic client, the `codex` process) with the real message
types where the SDK has them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from eval_harness.adapters import oneshot as mod
from eval_harness.adapters.base import NOT_A_PARAMETER
from eval_harness.config import ModelConfig

PRICED = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25}


def _cfg(provider: str, **kw: Any) -> ModelConfig:
    return ModelConfig.model_validate({"key": "k", "provider": provider, "model": "m-1", **kw})


# ------------------------------------------------------------------------- claude-code


def _fake_query(sent: dict[str, Any], *, is_error: bool = False) -> Any:
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    async def query(*, prompt: str, options: Any) -> Any:
        sent["prompt"] = prompt
        sent["options"] = options
        yield AssistantMessage(content=[TextBlock(text="draft")], model="m-1")
        yield ResultMessage(
            subtype="error_during_execution" if is_error else "success",
            duration_ms=10,
            duration_api_ms=9,
            is_error=is_error,
            num_turns=1,
            session_id="s",
            total_cost_usd=0.042,
            usage={"input_tokens": 1000, "output_tokens": 2000},
            result="rate limited" if is_error else "the summary",
        )

    return query


async def test_claude_code_reports_usage_and_sends_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claude_agent_sdk

    from eval_harness.adapters.claude_code import MAX_OUTPUT_TOKENS_ENV

    sent: dict[str, Any] = {}
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query(sent))
    result = await mod.oneshot("write it", system="sys", model=_cfg("claude-code"), max_tokens=2000)
    assert result.text == "the summary"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (1000, 2000)
    assert result.cost_usd == pytest.approx(0.042), "unpriced: the CLI's own figure"
    assert sent["options"].env[MAX_OUTPUT_TOKENS_ENV] == "2000"
    assert sent["options"].tools == [] and sent["options"].setting_sources == []
    assert result.request_settings["max_output_tokens"] == 2000
    assert result.latency_seconds >= 0


async def test_claude_code_prices_from_models_yaml_when_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query({}))
    result = await mod.oneshot(
        "x", system="s", model=_cfg("claude-code", price_per_mtok=PRICED), max_tokens=2000
    )
    assert result.cost_usd == pytest.approx(1000 * 5 / 1e6 + 2000 * 25 / 1e6)
    assert result.request_settings["cost_usd_reported"] == pytest.approx(0.042)


async def test_a_tiny_claude_code_ceiling_is_raised_to_the_floor_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classifier asks for 20; Claude Code's own thinking budget would fail at 20."""
    import claude_agent_sdk

    from eval_harness.adapters.claude_code import MAX_OUTPUT_TOKENS_ENV

    sent: dict[str, Any] = {}
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query(sent))
    result = await mod.oneshot("x", system="s", model=_cfg("claude-code"), max_tokens=20)
    floor = mod.CLAUDE_CODE_MIN_OUTPUT_TOKENS
    assert sent["options"].env[MAX_OUTPUT_TOKENS_ENV] == str(floor)
    assert result.request_settings["max_output_tokens"] == floor
    assert result.request_settings["max_output_tokens_requested"] == 20


async def test_claude_code_error_result_is_never_returned_as_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query({}, is_error=True))
    with pytest.raises(RuntimeError, match="rate limited"):
        await mod.oneshot("x", system="s", model=_cfg("claude-code"))


# ------------------------------------------------------------------------- anthropic


class _Usage:
    input_tokens = 300
    output_tokens = 120
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Block:
    type = "text"
    text = "hello"


class _Message:
    content: ClassVar[list[_Block]] = [_Block()]
    usage = _Usage()
    stop_reason = "max_tokens"


class _Messages:
    sent: ClassVar[dict[str, Any]] = {}

    async def create(self, **params: Any) -> _Message:
        _Messages.sent = params
        return _Message()


class _Anthropic:
    def __init__(self, **kw: Any) -> None:
        self.messages = _Messages()


async def test_anthropic_reports_usage_cost_and_stop_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Anthropic)
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv(mod.LEDGER_ENV, str(ledger))
    result = await mod.oneshot(
        "x", system="s", model=_cfg("anthropic", price_per_mtok=PRICED), max_tokens=777
    )
    assert result.text == "hello"
    assert _Messages.sent["max_tokens"] == 777
    assert result.stop_reason == "max_tokens", "a truncated answer must be visible as one"
    assert result.cost_usd == pytest.approx(300 * 5 / 1e6 + 120 * 25 / 1e6)
    # Before OneshotResult this branch never reached the ledger, so a priced anthropic judge
    # spent outside the limit it was configured under.
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows and rows[0]["cost_usd"] == pytest.approx(result.cost_usd)


async def test_the_limit_stops_every_provider_not_only_openai_compat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import anthropic

    _Messages.sent = {}
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Anthropic)
    ledger = tmp_path / "spend.jsonl"
    ledger.write_text(json.dumps({"cost_usd": 3.0}) + "\n")
    monkeypatch.setenv(mod.LEDGER_ENV, str(ledger))
    monkeypatch.setenv(mod.LIMIT_ENV, "2")
    with pytest.raises(mod.SpendLimitReached):
        await mod.oneshot("x", system="s", model=_cfg("anthropic"))
    assert _Messages.sent == {}


# ------------------------------------------------------------------------- codex-cli

CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "codex answer"}},
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 900, "cached_input_tokens": 100, "output_tokens": 1800},
    },
]


class _Proc:
    def __init__(self, out: list[dict[str, Any]], code: int = 0) -> None:
        self._out = "\n".join(json.dumps(e) for e in out).encode()
        self.returncode = code

    async def communicate(self, data: bytes | None = None) -> tuple[bytes, bytes]:
        return self._out, b"stderr tail"

    def kill(self) -> None:
        pass


def _fake_exec(seen: dict[str, Any], events: list[dict[str, Any]], code: int = 0) -> Any:
    async def create(*cmd: str, **kw: Any) -> _Proc:
        seen["cmd"] = list(cmd)
        seen["cwd"] = kw["cwd"]
        seen["cwd_entries"] = sorted(p.name for p in Path(kw["cwd"]).iterdir())
        return _Proc(events, code)

    return create


async def test_codex_one_shot_is_hermetic_and_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec(seen, CODEX_EVENTS))
    result = await mod.oneshot(
        "the prompt", system="the system", model=_cfg("codex-cli"), max_tokens=2000
    )
    cmd = seen["cmd"]
    assert cmd[:2] == ["codex", "exec"]
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--skip-git-repo-check" in cmd and "--ignore-user-config" in cmd
    assert cmd[cmd.index("-m") + 1] == "m-1", "the model is pinned, not ~/.codex's default"
    assert cmd[cmd.index("-C") + 1] == seen["cwd"]
    assert seen["cwd_entries"] == [], "the working root is empty: no AGENTS.md, no repository"
    instructions = next(c for c in cmd if c.startswith("model_instructions_file="))
    assert seen["cwd"] not in instructions, "the system prompt file sits outside the root"
    assert result.text == "codex answer"
    assert (result.usage.input_tokens, result.usage.cache_read_tokens) == (800, 100)
    assert result.usage.output_tokens == 1800
    assert result.request_settings["max_output_tokens"] == NOT_A_PARAMETER
    assert result.request_settings["max_output_tokens_requested"] == 2000


async def test_codex_failure_raises_instead_of_returning_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [{"type": "turn.failed", "error": {"message": "usage limit"}}]
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec({}, events, code=1))
    with pytest.raises(RuntimeError, match="usage limit"):
        await mod.oneshot("x", system="s", model=_cfg("codex-cli"))


async def test_codex_one_shot_that_ran_a_command_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
        *CODEX_EVENTS,
    ]
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec({}, events))
    with pytest.raises(RuntimeError, match="ran 1 command"):
        await mod.oneshot("x", system="s", model=_cfg("codex-cli"))
