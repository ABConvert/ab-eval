"""The judge must work on any OpenAI-shaped endpoint, and must never leak the key."""

from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from eval_harness.adapters import oneshot as mod
from eval_harness.config import ModelConfig


def _cfg(**kw: object) -> ModelConfig:
    base = {
        "key": "gemini-judge",
        "provider": "openai-compat",
        "model": "gemini-3.8-flash",
        "base_url": "https://example.invalid/v1",
        "api_key_env": "JUDGE_KEY",
    }
    base.update(kw)
    return ModelConfig.model_validate(base)


class _Client:
    """Stands in for httpx.AsyncClient, capturing the one request the judge makes."""

    sent: ClassVar[dict[str, object]] = {}
    reply: ClassVar[tuple[int, dict[str, object]]] = (200, {})

    def __init__(self, **kw: object) -> None:
        pass

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> object:
        _Client.sent = {"url": url, "headers": headers, "body": json}
        code, payload = _Client.reply
        return httpx.Response(code, json=payload)


@pytest.fixture(autouse=True)
def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("JUDGE_KEY", "sk-secret-judge-key")
    _Client.reply = (200, {"choices": [{"message": {"content": "0.7 because ..."}}]})


async def test_openai_compat_judge_returns_the_content() -> None:
    text = await mod.oneshot("rate this", system="be a judge", model=_cfg(), max_tokens=400)
    assert text == "0.7 because ..."
    assert _Client.sent["url"] == "https://example.invalid/v1/chat/completions"
    body = _Client.sent["body"]
    assert body["messages"][0] == {"role": "system", "content": "be a judge"}
    assert body["messages"][1] == {"role": "user", "content": "rate this"}
    assert body["max_tokens"] == 400


async def test_reasoning_field_is_ignored_in_favour_of_content() -> None:
    """Thinking models return both; the answer is content."""
    _Client.reply = (200, {"choices": [{"message": {"reasoning": "hmm", "content": "0.4"}}]})
    assert await mod.oneshot("x", system="s", model=_cfg()) == "0.4"


async def test_an_error_response_never_carries_the_key() -> None:
    _Client.reply = (401, {"error": "bad key"})
    with pytest.raises(RuntimeError) as e:
        await mod.oneshot("x", system="s", model=_cfg())
    assert "sk-secret-judge-key" not in str(e.value)
    assert "401" in str(e.value)


async def test_a_missing_key_says_which_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUDGE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JUDGE_KEY is not set"):
        await mod.oneshot("x", system="s", model=_cfg())


async def test_a_local_endpoint_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama and friends ignore the bearer; requiring one would rule them out."""
    _Client.reply = (200, {"choices": [{"message": {"content": "ok"}}]})
    assert await mod.oneshot("x", system="s", model=_cfg(api_key_env=None)) == "ok"
    assert _Client.sent["headers"] == {}


async def test_unknown_provider_still_refuses() -> None:
    with pytest.raises(KeyError):
        await mod.oneshot("x", system="s", model=_cfg(provider="carrier-pigeon"))


async def test_a_judge_that_declares_nothing_sends_exactly_what_it_used_to() -> None:
    """The judge scores every published number, so its request must not drift by accident."""
    await mod.oneshot("x", system="s", model=_cfg(), max_tokens=400)
    assert set(_Client.sent["body"]) == {"model", "max_tokens", "messages"}  # type: ignore[arg-type]


async def test_a_judge_that_declares_reasoning_and_temperature_has_them_sent() -> None:
    """`judge_config` is snapshotted into every summary.json; it has to be what was sent."""
    await mod.oneshot(
        "x",
        system="s",
        model=_cfg(reasoning_param="reasoning_effort", effort="medium", temperature=0.0),
        max_tokens=400,
    )
    body = _Client.sent["body"]
    assert body["reasoning_effort"] == "medium"  # type: ignore[index]
    assert body["temperature"] == 0.0, "0.0 is a setting, not an absence"  # type: ignore[index]


async def test_a_judge_endpoint_that_rejects_the_reasoning_field_says_so() -> None:
    _Client.reply = (400, {"error": "unknown field"})
    with pytest.raises(RuntimeError) as caught:
        await mod.oneshot("x", system="s", model=_cfg(reasoning_param="reasoning.effort"))
    message = str(caught.value)
    assert "reasoning" in message and "reasoning_param: omit" in message
    assert "https://example.invalid/v1/chat/completions" in message
    assert "sk-secret-judge-key" not in message


# ------------------------------------------------------------------ the judge's spend ledger

PRICED = {"input": 0.75, "output": 3.75, "cache_read": 0.075, "cache_write": 0.0}


def _usage_reply(prompt: int, completion: int) -> tuple[int, dict[str, object]]:
    return (
        200,
        {
            "choices": [{"message": {"content": "0.8"}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


async def test_each_priced_judge_call_is_written_to_the_ledger(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-scoring ten runs is ten processes; the ledger is how their spend adds up."""
    import json
    from pathlib import Path

    ledger = Path(str(tmp_path)) / "spend.jsonl"
    monkeypatch.setenv("ABEVAL_SPEND_LEDGER", str(ledger))
    _Client.reply = _usage_reply(1_000_000, 100_000)
    await mod.oneshot("rate", system="judge", model=_cfg(price_per_mtok=PRICED), max_tokens=10)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["model"] == "gemini-3.8-flash"
    assert rows[0]["cost_usd"] == pytest.approx(0.75 + 0.375)


async def test_a_call_that_would_pass_the_limit_is_never_sent(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit is checked before the request, so a stopped run spends nothing more."""
    import json
    from pathlib import Path

    ledger = Path(str(tmp_path)) / "spend.jsonl"
    ledger.write_text(json.dumps({"model": "x", "cost_usd": 20.0}) + "\n")
    monkeypatch.setenv("ABEVAL_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("ABEVAL_SPEND_LIMIT_USD", "20")
    _Client.sent = {}
    with pytest.raises(mod.SpendLimitReached, match="20"):
        await mod.oneshot("rate", system="judge", model=_cfg(price_per_mtok=PRICED), max_tokens=10)
    assert _Client.sent == {}


async def test_no_ledger_configured_behaves_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ABEVAL_SPEND_LEDGER", raising=False)
    monkeypatch.delenv("ABEVAL_SPEND_LIMIT_USD", raising=False)
    _Client.reply = _usage_reply(10, 5)
    text = await mod.oneshot(
        "rate", system="judge", model=_cfg(price_per_mtok=PRICED), max_tokens=10
    )
    assert text == "0.8"


def test_a_limit_without_a_ledger_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A limit with nowhere to count against would silently never trip."""
    monkeypatch.delenv("ABEVAL_SPEND_LEDGER", raising=False)
    monkeypatch.setenv("ABEVAL_SPEND_LIMIT_USD", "20")
    with pytest.raises(RuntimeError, match="ABEVAL_SPEND_LEDGER"):
        mod.spent_so_far()


def test_thinking_left_out_of_completion_tokens_is_still_counted_as_output() -> None:
    """Gemini's OpenAI endpoint reports thinking only in total_tokens, and bills it as output.

    Measured on gemini-3.8-flash: prompt 21, completion 1, total 153. Counting completion
    alone would put 131 billed tokens nowhere and let a spend limit never trip.
    """
    from eval_harness.adapters.openai_compat import usage_from_response

    u = usage_from_response({"prompt_tokens": 21, "completion_tokens": 1, "total_tokens": 153})
    assert u.input_tokens == 21
    assert u.output_tokens == 132


def test_an_endpoint_whose_total_adds_up_is_unchanged() -> None:
    from eval_harness.adapters.openai_compat import usage_from_response

    u = usage_from_response({"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140})
    assert u.output_tokens == 40
    assert usage_from_response({"prompt_tokens": 10, "completion_tokens": 4}).output_tokens == 4
