from typing import Any

from pydantic import BaseModel

from eval_harness.config import ModelConfig
from eval_harness.metrics.judge import SubscriptionJudge, extract_json


class ReasonScore(BaseModel):
    score: float
    reason: str


def _cfg() -> ModelConfig:
    return ModelConfig(
        key="judge", provider="claude-code", model="claude-fable-5-1", effort="medium"
    )


async def test_generate_with_schema_parses_json_and_retries_once() -> None:
    replies = ["not json at all", 'Sure: {"score": 0.8, "reason": "fine"}']
    calls: list[str] = []

    async def fake(prompt: str, *, system: str, model: Any, max_tokens: int) -> str:
        calls.append(prompt)
        return replies.pop(0)

    judge = SubscriptionJudge(_cfg(), generate_fn=fake)
    out = await judge.a_generate_with_schema("rate this", schema=ReasonScore)
    assert isinstance(out, ReasonScore) and out.score == 0.8
    assert len(calls) == 2 and "exactly these keys: ['score', 'reason']" in calls[0]
    assert "not valid JSON" in calls[1]


def test_extract_json_and_model_name() -> None:
    assert extract_json('prefix {"a": 1} suffix') == {"a": 1}
    judge = SubscriptionJudge(_cfg(), generate_fn=lambda *a, **k: "x")
    assert judge.get_model_name().startswith("claude-fable-5-1 (claude-code")
