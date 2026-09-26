"""One-turn, tool-free model calls for the classifier and the judge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


def _record(model: ModelConfig, raw_usage: dict[str, Any] | None) -> None:
    ledger = _ledger()
    if ledger is None or model.price_per_mtok is None:
        return
    from eval_harness.adapters.openai_compat import usage_from_response
    from eval_harness.adapters.pricing import cost_usd

    usage = usage_from_response(raw_usage)
    row = {
        "model": model.model,
        "cost_usd": cost_usd(usage, model.price_per_mtok),
        **usage.model_dump(),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def oneshot(prompt: str, *, system: str, model: ModelConfig, max_tokens: int = 500) -> str:
    if model.provider == "claude-code":
        # `max_tokens` is NOT applied on this branch, deliberately, and that is a hole worth
        # knowing about: the judge asks for its configured 4000 and the classifier for 20, and
        # a claude-code judge or classifier gets neither. The agent loop in claude_code.py
        # carries the ceiling through CLAUDE_CODE_MAX_OUTPUT_TOKENS on `options.env`, and the
        # same line would work here — except that the classifier's 20 would then become a real
        # 20-token ceiling, and Claude Code asks for a thinking budget of its own within that
        # limit, so the call would likely fail outright rather than return a one-word label.
        # Honouring it here means first deciding what the classifier should actually ask for.
        # Until then this says so out loud instead of looking like it was honoured. The
        # openai-compat branch below has no such excuse and applies everything it is given.
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query

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
        )
        text = ""
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
        return text
    if model.provider == "anthropic":
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
        return "\n".join(str(b.text) for b in content if getattr(b, "type", "") == "text")
    if model.provider == "openai-compat":
        import httpx

        from eval_harness.adapters.openai_compat import (
            reasoning_body,
            rejected_the_reasoning_field,
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
        _check_limit(model)
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
        _record(model, data.get("usage"))
        message = ((data.get("choices") or [{}])[0].get("message")) or {}
        # Reasoning models put their thinking in a sibling field; the answer is `content`.
        return str(message.get("content") or "")
    raise KeyError(f"provider {model.provider!r} does not support one-shot calls")
