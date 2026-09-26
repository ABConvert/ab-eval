"""Agent loop against any OpenAI-compatible /chat/completions endpoint.

Used for DeepSeek, and for anything else that speaks the same shape. The harness owns the
loop here, exactly as it does for the `anthropic` provider: the model only ever sees our
sandbox tools, and every tool call runs in the container.

The key comes from an environment variable named in config/models.yaml, never from the
config file itself.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from eval_harness.adapters.base import AgentResult, Caps, ToolExecutor, ToolSpec, Usage
from eval_harness.adapters.pricing import cost_usd
from eval_harness.adapters.tracing import Tracer
from eval_harness.config import ModelConfig

DEFAULT_BASE_URL = "https://api.deepseek.com"
# One request may take as long as the case itself is allowed: Caps.wall_clock_seconds
# decides when to stop, and a shorter client timeout just aborts a case the caps still
# permit. Local weights at ~17 tokens/second can spend more than ten minutes on a turn.
REQUEST_TIMEOUT = 1800.0


def reasoning_body(param: str, effort: str) -> dict[str, Any]:
    """The body fragment that tells this endpoint how hard to think, which may be nothing.

    OpenAI-compatible is a shape, not a contract: hosted APIs and most gateways read
    `reasoning_effort`, some read a `reasoning` object, and a plain llama.cpp or ollama shim
    reads neither and answers 400 to a field it does not know. Guessing costs a whole run, so
    the model says which in `reasoning_param` and the default sends nothing — which is what
    every endpoint got before this existed, and what the two bench-v2 openai-compat runs got
    while their config said `effort: high`.
    """
    if param == "reasoning_effort":
        return {"reasoning_effort": effort}
    if param == "reasoning.effort":
        return {"reasoning": {"effort": effort}}
    return {}


def rejected_the_reasoning_field(
    *, status: int, body: str, sent: dict[str, Any], base_url: str, model: str, param: str
) -> str:
    """What to say when a request carrying a reasoning field comes back 4xx.

    Deliberately not a claim about cause — the server may have rejected the field, or the
    prompt, or the key. It names what this harness added to the request and how to stop
    adding it, because that is the one thing the reader cannot see from a raw status dump.
    """
    return (
        f"{model} at {base_url}/chat/completions returned {status}. This request carried "
        f"{json.dumps(sent)}, which the harness adds because {model} is configured "
        f"reasoning_param: {param}. If this endpoint does not accept that field, set "
        f"reasoning_param: omit in config/models.yaml. Server said: {body[:400]}"
    )


def tool_schema(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def usage_from_response(raw: dict[str, Any] | None) -> Usage:
    """Map OpenAI-style usage, keeping cache hits separate from fresh input."""
    u = raw or {}
    cached = int(u.get("prompt_cache_hit_tokens") or 0)
    if not cached:
        details = u.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
    prompt = int(u.get("prompt_tokens") or 0)
    completion = int(u.get("completion_tokens") or 0)
    # Gemini's OpenAI endpoint leaves thinking out of completion_tokens and reports it only in
    # total_tokens, though it bills it as output. Whatever the total holds beyond prompt and
    # completion is therefore output; for an endpoint whose total adds up this is a no-op.
    completion = max(completion, int(u.get("total_tokens") or 0) - prompt)
    return Usage(
        # prompt_tokens includes the cached part, so subtract it to avoid counting twice
        input_tokens=max(prompt - cached, 0),
        output_tokens=completion,
        cache_read_tokens=cached,
        cache_write_tokens=0,  # this API reports no cache-creation tokens
    )


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string, and a model can still get that wrong."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__malformed_arguments__": raw[:2000]}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


class OpenAICompatAdapter:
    """Harness-owned tool loop over an OpenAI-compatible chat completions API."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.name = cfg.key
        self.base_url = str(getattr(cfg, "base_url", None) or DEFAULT_BASE_URL).rstrip("/")
        env_var = str(getattr(cfg, "api_key_env", None) or "DEEPSEEK_API_KEY")
        self.api_key_env = env_var
        key = os.environ.get(env_var)
        if not key:
            raise RuntimeError(
                f"{env_var} is not set; export it before running {cfg.key!r} "
                "(keys never live in config/models.yaml)"
            )
        self._key = key

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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        schema = tool_schema(tools)
        usage = Usage()
        turns = tool_calls = 0
        cap_hit: str | None = None
        stop_reason = "none"
        final_text = ""
        transcript: list[dict[str, Any]] = []
        started = time.monotonic()
        # Built once and recorded as-is: what the trace and the run artifact say was sent
        # has to be the same object that is sent, or the artifact is a second guess at it.
        reasoning = reasoning_body(self.cfg.reasoning_param, self.cfg.effort)
        extra: dict[str, Any] = dict(reasoning)
        if self.cfg.temperature is not None:
            extra["temperature"] = self.cfg.temperature
        settings: dict[str, Any] = {
            "provider": "openai-compat",
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_output_tokens,
            "temperature": self.cfg.temperature,
            "reasoning_param": self.cfg.reasoning_param,
            # The effort the endpoint was actually told, or None when it was told nothing.
            # `effort` in config is not evidence it crossed the wire; this is.
            "reasoning_sent": reasoning or None,
            "builtin_tools": "none",
        }

        params: dict[str, Any] = {"max_turns": caps.max_turns, **extra}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            with trace.generation(
                "openai-compat", model=self.cfg.model, model_parameters=params
            ) as gen:
                while True:
                    if turns >= caps.max_turns:
                        cap_hit, stop_reason = "max_turns", "max_turns"
                        break
                    if time.monotonic() - started > caps.wall_clock_seconds:
                        cap_hit, stop_reason = "wall_clock", "wall_clock"
                        break
                    if usage.output_tokens > caps.max_output_tokens_total:
                        cap_hit, stop_reason = "max_output_tokens", "max_output_tokens"
                        break

                    body: dict[str, Any] = {
                        "model": self.cfg.model,
                        "messages": messages,
                        "max_tokens": self.cfg.max_output_tokens,
                        **extra,
                    }
                    if schema:
                        body["tools"] = schema
                    reply = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._key}"},
                        json=body,
                    )
                    if reply.status_code >= 400:
                        # Never let the key reach a log or a record; reply.text is the
                        # server's own body, which never contains the Authorization header.
                        if reasoning:
                            raise RuntimeError(
                                rejected_the_reasoning_field(
                                    status=reply.status_code,
                                    body=reply.text,
                                    sent=reasoning,
                                    base_url=self.base_url,
                                    model=self.cfg.model,
                                    param=self.cfg.reasoning_param,
                                )
                            )
                        raise RuntimeError(
                            f"{self.cfg.model} returned {reply.status_code}: {reply.text[:400]}"
                        )
                    data = reply.json()
                    usage.add(usage_from_response(data.get("usage")))
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    turns += 1
                    text = str(message.get("content") or "")
                    calls = message.get("tool_calls") or []
                    transcript.append(
                        {"role": "assistant", "content": text, "tool_calls": len(calls)}
                    )
                    if text:
                        final_text = text

                    if not calls:
                        stop_reason = str(choice.get("finish_reason") or "stop")
                        break

                    # Echo the assistant turn back verbatim, then answer each call.
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.get("content"),
                            "tool_calls": calls,
                        }
                    )
                    for call in calls:
                        fn = call.get("function") or {}
                        name = str(fn.get("name") or "")
                        args = parse_arguments(fn.get("arguments"))
                        result = await execute(name, args)
                        tool_calls += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id"),
                                "content": result.content[:60_000],
                            }
                        )

                priced = cost_usd(usage, self.cfg.price_per_mtok)
                gen.end(
                    input={"task_chars": len(task)},
                    output={"stop_reason": stop_reason, "turns": turns},
                    usage=usage,
                    cost=priced,
                )

        settings["cost_usd_from_prices"] = priced
        settings["wall_clock_seconds"] = round(time.monotonic() - started, 1)
        return AgentResult(
            turns=turns,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=priced,
            stop_reason=stop_reason,
            cap_hit=cap_hit,
            final_text=final_text,
            transcript=transcript,
            request_settings=settings,
        )
