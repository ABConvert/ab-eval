"""One-turn, tool-free model calls: the classifier, the judges, and prompt-task candidates.

A prompt task benchmarks one completion per case, so the call has to report what it cost and
how long it took, not only what it said. `oneshot` returns an `OneshotResult` for that reason;
callers that only want the words take `.text`.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval_harness.adapters.base import NOT_A_PARAMETER, Usage
from eval_harness.adapters.pricing import cost_usd
from eval_harness.config import ModelConfig

# A judge call is one request; the cap matches the agent loop's, so a slow endpoint is not
# mistaken for a broken one.
REQUEST_TIMEOUT = 600.0

# A re-score is one process per run, so a budget across ten runs cannot live in memory. Each
# priced call appends its cost to a JSONL ledger, and the limit is checked against the whole
# ledger before the next request goes out. Both are opt-in; with neither set, a call is the
# request it always was.
LEDGER_ENV = "ABEVAL_SPEND_LEDGER"
LIMIT_ENV = "ABEVAL_SPEND_LIMIT_USD"


# Claude Code asks for a thinking budget of its own inside CLAUDE_CODE_MAX_OUTPUT_TOKENS, so a
# ceiling as small as the classifier's 20 would fail the call rather than shorten the answer.
# Below this floor the floor is sent instead, and request_settings says so.
CLAUDE_CODE_MIN_OUTPUT_TOKENS = 1024


class OneshotResult(BaseModel):
    """One completion and what it cost. `request_settings` is what was sent, not what was asked."""

    text: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    request_settings: dict[str, Any] = Field(default_factory=dict)
    # The provider's own reason for stopping (`end_turn`, `max_tokens`, `length`), where it
    # gives one: a truncated summary should be told apart from a short one.
    stop_reason: str | None = None


class SpendLimitReached(RuntimeError):
    """The ledger has reached ABEVAL_SPEND_LIMIT_USD; the call was not made."""


def _ledger() -> Path | None:
    raw = os.environ.get(LEDGER_ENV)
    return Path(raw) if raw else None


def spent_so_far() -> float:
    """Total USD in the ledger. A limit with no ledger to count against is an error."""
    ledger = _ledger()
    if ledger is None:
        if os.environ.get(LIMIT_ENV):
            raise RuntimeError(f"{LIMIT_ENV} is set but {LEDGER_ENV} is not; nothing to count")
        return 0.0
    if not ledger.exists():
        return 0.0
    return sum(
        float(json.loads(line).get("cost_usd") or 0.0)
        for line in ledger.read_text().splitlines()
        if line.strip()
    )


def _check_limit(model: ModelConfig) -> None:
    limit = os.environ.get(LIMIT_ENV)
    spent = spent_so_far()
    if limit and spent >= float(limit):
        raise SpendLimitReached(
            f"spend ledger is at ${spent:.2f}, at or over the ${float(limit):.2f} limit; "
            f"{model.model} was not called"
        )


def _record(model: ModelConfig, result: OneshotResult) -> None:
    """Append a priced call to the ledger. Calls that cost nothing are not rows."""
    ledger = _ledger()
    if ledger is None or result.cost_usd <= 0:
        return
    row = {"model": model.model, "cost_usd": result.cost_usd, **result.usage.model_dump()}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def oneshot(
    prompt: str, *, system: str, model: ModelConfig, max_tokens: int = 500
) -> OneshotResult:
    """One request, no tools. The spend limit is checked before it and the ledger after it."""
    _check_limit(model)
    started = time.monotonic()
    if model.provider == "claude-code":
        result = await _claude_code(prompt, system=system, model=model, max_tokens=max_tokens)
    elif model.provider == "anthropic":
        result = await _anthropic(prompt, system=system, model=model, max_tokens=max_tokens)
    elif model.provider == "openai-compat":
        result = await _openai_compat(prompt, system=system, model=model, max_tokens=max_tokens)
    elif model.provider == "codex-cli":
        result = await _codex_cli(prompt, system=system, model=model, max_tokens=max_tokens)
    else:
        raise KeyError(f"provider {model.provider!r} does not support one-shot calls")
    result.latency_seconds = round(time.monotonic() - started, 3)
    _record(model, result)
    return result


async def _claude_code(
    prompt: str, *, system: str, model: ModelConfig, max_tokens: int
) -> OneshotResult:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query

    from eval_harness.adapters.claude_code import MAX_OUTPUT_TOKENS_ENV, usage_from_result

    # The ceiling travels the same way the agent loop sends it. A request under the floor gets
    # the floor (see CLAUDE_CODE_MIN_OUTPUT_TOKENS), and the record names both numbers.
    sent = max(max_tokens, CLAUDE_CODE_MIN_OUTPUT_TOKENS)
    os.environ.pop("CLAUDECODE", None)
    options = ClaudeAgentOptions(
        model=model.model,
        effort=model.effort,  # type: ignore[arg-type]
        max_turns=1,
        system_prompt=system,
        setting_sources=[],
        tools=[],
        allowed_tools=[],
        permission_mode="dontAsk",
        strict_mcp_config=True,
        env={MAX_OUTPUT_TOKENS_ENV: str(sent)},
    )
    text = ""
    usage = Usage()
    reported: float | None = None
    stop: str | None = None
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            blocks: list[Any] = list(msg.content)
            text = "\n".join(str(b.text) for b in blocks if hasattr(b, "text")) or text
        elif isinstance(msg, ResultMessage):
            if msg.is_error:
                # Rate limits and auth failures arrive as an error result whose text is
                # the CLI's message; never return that as if it were the model's answer.
                raise RuntimeError(f"claude-code error result ({msg.subtype}): {msg.result}")
            if msg.result:
                text = msg.result
            usage = usage_from_result(msg.usage)
            reported = msg.total_cost_usd
            stop = getattr(msg, "stop_reason", None) or msg.subtype
    priced = cost_usd(usage, model.price_per_mtok)
    return OneshotResult(
        text=text,
        usage=usage,
        # Prices in models.yaml win, as they do everywhere a cost is derived; the CLI's own
        # figure is the fallback, so an unpriced subscription model still reports something.
        cost_usd=priced if model.price_per_mtok is not None else float(reported or 0.0),
        stop_reason=stop,
        request_settings={
            "provider": "claude-code",
            "model": model.model,
            "effort": model.effort,
            "max_output_tokens": sent,
            "max_output_tokens_requested": max_tokens,
            "max_output_tokens_via": f"{MAX_OUTPUT_TOKENS_ENV} on the CLI the SDK spawns",
            "temperature": NOT_A_PARAMETER,
            "cost_usd_reported": reported,
            "builtin_tools": "disabled",
        },
    )


async def _anthropic(
    prompt: str, *, system: str, model: ModelConfig, max_tokens: int
) -> OneshotResult:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(max_retries=4)
    params: dict[str, Any] = {
        "model": model.model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": model.effort},
    }
    message = await client.messages.create(**params)
    content: list[Any] = list(message.content)
    raw = getattr(message, "usage", None)
    usage = Usage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(raw, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(raw, "cache_creation_input_tokens", 0) or 0),
    )
    return OneshotResult(
        text="\n".join(str(b.text) for b in content if getattr(b, "type", "") == "text"),
        usage=usage,
        cost_usd=cost_usd(usage, model.price_per_mtok),
        stop_reason=getattr(message, "stop_reason", None),
        request_settings={
            "provider": "anthropic",
            "model": model.model,
            "effort": model.effort,
            "max_output_tokens": max_tokens,
            "temperature": NOT_A_PARAMETER,
        },
    )


async def _openai_compat(
    prompt: str, *, system: str, model: ModelConfig, max_tokens: int
) -> OneshotResult:
    import httpx

    from eval_harness.adapters.openai_compat import (
        reasoning_body,
        rejected_the_reasoning_field,
        usage_from_response,
    )

    # Any OpenAI-shaped endpoint: a hosted API, or a local server. The judge should not be
    # locked to one vendor — least of all to a vendor whose models it is scoring.
    base_url = str(model.base_url or "").rstrip("/")
    if not base_url:
        raise KeyError(f"{model.key!r} needs base_url to be used for one-shot calls")
    env_var = model.api_key_env or ""
    key = os.environ.get(env_var) if env_var else None
    if env_var and not key:
        raise RuntimeError(
            f"{env_var} is not set; export it before using {model.key!r} "
            "(keys never live in config/models.yaml)"
        )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    # The same two fields the agent loop sends, on the same terms: `reasoning_param`
    # defaults to omit and `temperature` to unset, so a judge that declares neither is
    # one request identical to the one this sent before. A judge that declares them is
    # entitled to have them reach the server — models.example.yaml offers an
    # openai-compat judge, the /setup form offers it both boxes, and `judge_config` is
    # snapshotted into every summary.json, so a declared value that went nowhere would
    # be the whole defect again in the one entry that scores the others.
    reasoning = reasoning_body(model.reasoning_param, model.effort)
    body: dict[str, Any] = {
        "model": model.model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **reasoning,
    }
    if model.temperature is not None:
        body["temperature"] = model.temperature
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as http:
        reply = await http.post(f"{base_url}/chat/completions", headers=headers, json=body)
    if reply.status_code >= 400:
        # Never let the key reach a log or a record; reply.text is the server's own body.
        if reasoning:
            raise RuntimeError(
                rejected_the_reasoning_field(
                    status=reply.status_code,
                    body=reply.text,
                    sent=reasoning,
                    base_url=base_url,
                    model=model.model,
                    param=model.reasoning_param,
                )
            )
        raise RuntimeError(f"{model.model} returned {reply.status_code}: {reply.text[:400]}")
    data = reply.json()
    usage = usage_from_response(data.get("usage"))
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return OneshotResult(
        # Reasoning models put their thinking in a sibling field; the answer is `content`.
        text=str(message.get("content") or ""),
        usage=usage,
        cost_usd=cost_usd(usage, model.price_per_mtok),
        stop_reason=choice.get("finish_reason"),
        request_settings={
            "provider": "openai-compat",
            "model": model.model,
            "base_url": base_url,
            **{k: v for k, v in body.items() if k not in ("messages", "model")},
        },
    )


# Codex's own execution surfaces, switched off for a one-shot call: nothing here should run a
# command, browse, or read a project file. The read-only sandbox is the second fence.
CODEX_ONESHOT_DISABLED = [
    "shell_tool",
    "unified_exec",
    "apps",
    "browser_use",
    "image_generation",
    "computer_use",
    "collaboration_modes",
]


def codex_oneshot_command(model: ModelConfig, *, workdir: Path, instructions: Path) -> list[str]:
    """`codex exec` for one hermetic completion.

    Hermetic means: an empty working root (so no AGENTS.md or repository leaks into the
    prompt), the read-only sandbox, no user config or rules, no web search, the system prompt
    in place of Codex's built-in instructions, and the model pinned with `-m` so a changed
    default in ~/.codex cannot swap the candidate under a running benchmark.
    """
    from eval_harness.adapters.codex_cli import EFFORTS

    effort = model.effort if model.effort in EFFORTS else "high"
    cmd = [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-C",
        str(workdir),
        "-m",
        model.model,
    ]
    for feature in CODEX_ONESHOT_DISABLED:
        cmd += ["--disable", feature]
    for override in (
        f"model_reasoning_effort={json.dumps(effort)}",
        'web_search="disabled"',
        "tools.view_image=false",
        "features.hooks=false",
        "project_doc_max_bytes=0",
        f"model_instructions_file={json.dumps(str(instructions))}",
    ):
        cmd += ["-c", override]
    cmd.append("-")  # prompt on stdin
    return cmd


async def _codex_cli(
    prompt: str, *, system: str, model: ModelConfig, max_tokens: int
) -> OneshotResult:
    from eval_harness.adapters.codex_cli import parse_events, usage_from_events

    with tempfile.TemporaryDirectory(prefix="abeval-oneshot-") as tmp:
        root = Path(tmp)
        workdir = root / "work"
        workdir.mkdir()
        # Outside the working root, so the model's read-only view of its cwd is empty.
        instructions = root / "instructions.md"
        instructions.write_text(system)
        cmd = codex_oneshot_command(model, workdir=workdir, instructions=instructions)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workdir),
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=REQUEST_TIMEOUT
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(f"codex exec gave no answer within {REQUEST_TIMEOUT:.0f}s") from None
    parsed = parse_events(out.decode(errors="replace").splitlines())
    if proc.returncode != 0 or parsed["failed"]:
        tail = err.decode(errors="replace")[-800:]
        raise RuntimeError(
            f"codex exec exited {proc.returncode}: {parsed['failed'] or ''} {tail}".strip()
        )
    if parsed["host_exec"]:
        # A one-shot answer that ran something is not a one-shot answer; refuse to score it.
        raise RuntimeError(f"codex exec ran {len(parsed['host_exec'])} command(s) in a one-shot")
    usage = usage_from_events(parsed["usage"])
    return OneshotResult(
        text=parsed["final_text"],
        usage=usage,
        cost_usd=cost_usd(usage, model.price_per_mtok),
        stop_reason="end_turn",
        request_settings={
            "provider": "codex-cli",
            "model": model.model,
            "effort": model.effort,
            # `codex exec` has no output ceiling to set (its max_output_tokens keys budget tool
            # output) and no temperature; the requested ceiling is kept so a reader can see it.
            "max_output_tokens": NOT_A_PARAMETER,
            "max_output_tokens_requested": max_tokens,
            "temperature": NOT_A_PARAMETER,
            "sandbox": "read-only, empty working root, user config and rules ignored",
            "builtin_tools": "disabled: " + ", ".join(CODEX_ONESHOT_DISABLED),
        },
    )
