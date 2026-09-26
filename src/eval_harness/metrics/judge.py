"""A DeepEval judge model backed by the harness's one-shot helper (subscription or API)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM

from eval_harness.adapters.oneshot import oneshot
from eval_harness.config import ModelConfig

JSON_SYSTEM = (
    "You are an exacting code reviewer acting as an evaluation judge. Follow the instructions "
    "exactly. When asked for JSON, reply with a single JSON object and nothing else."
)


def run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even when DeepEval calls us inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    data: dict[str, Any] = json.loads(m.group(0))
    return data


class SubscriptionJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """G-Eval calls `a_generate_with_schema(prompt, schema=...)` on custom judge models."""

    def __init__(self, cfg: ModelConfig, generate_fn: Any | None = None) -> None:
        self.cfg = cfg
        self._generate = generate_fn or oneshot
        super().__init__(cfg.model)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return self.cfg.model

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"{self.cfg.model} ({self.cfg.provider}, effort={self.cfg.effort})"

    async def _ask(self, prompt: str) -> str:
        text: str = await self._generate(
            prompt, system=JSON_SYSTEM, model=self.cfg, max_tokens=self.cfg.max_output_tokens
        )
        return text

    async def a_generate(self, *args: Any, **kwargs: Any) -> str:
        prompt = str(args[0]) if args else str(kwargs.get("prompt", ""))
        return await self._ask(prompt)

    def generate(self, *args: Any, **kwargs: Any) -> str:
        result: str = run_sync(self.a_generate(*args, **kwargs))
        return result

    async def a_generate_with_schema(self, *args: Any, schema: Any = None, **kwargs: Any) -> Any:
        prompt = str(args[0]) if args else str(kwargs.get("prompt", ""))
        if schema is None:
            return await self._ask(prompt)
        fields = list(getattr(schema, "model_fields", {}).keys())
        instruction = (
            f"\n\nRespond with a single JSON object with exactly these keys: {fields}. No prose."
        )
        text = await self._ask(prompt + instruction)
        try:
            return schema.model_validate(extract_json(text))
        except Exception:
            retry = " Your previous reply was not valid JSON for that schema."
            text = await self._ask(prompt + instruction + retry)
            return schema.model_validate(extract_json(text))

    def generate_with_schema(self, *args: Any, schema: Any = None, **kwargs: Any) -> Any:
        return run_sync(self.a_generate_with_schema(*args, schema=schema, **kwargs))
