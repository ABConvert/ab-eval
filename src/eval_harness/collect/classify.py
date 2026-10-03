from __future__ import annotations

import asyncio
import re

from eval_harness.adapters.oneshot import oneshot
from eval_harness.collect.sanitize import sanitize_text
from eval_harness.config import ModelConfig

LABEL_KINDS = {
    "bug": "bug_fix",
    "feature": "feature",
    "improvement": "feature",
    "enhancement": "feature",
}

SYSTEM = (
    "You classify engineering tickets. Answer with exactly one token: bug_fix if the ticket "
    "reports behaviour that is wrong and asks for it to be corrected, or feature if it asks "
    "for new or changed behaviour, a refactor, or an improvement. No other words."
)


def kind_from_labels(labels: list[str]) -> str | None:
    for label in labels:
        kind = LABEL_KINDS.get(label.strip().lower())
        if kind:
            return kind
    return None


def parse_kind(text: str) -> str:
    m = re.search(r"\b(bug_fix|feature)\b", text.lower())
    if not m:
        raise ValueError(f"classifier returned no kind: {text[:200]!r}")
    return m.group(1)


async def classify_kind_async(title: str, description: str, *, model: ModelConfig) -> str:
    title, description = sanitize_text(title), sanitize_text(description)
    prompt = f"Title: {title}\n\nDescription:\n{description[:6000]}\n\nAnswer: bug_fix or feature"
    return parse_kind(await oneshot(prompt, system=SYSTEM, model=model, max_tokens=20))


def classify_kind(title: str, description: str, *, model: ModelConfig) -> str:
    return asyncio.run(classify_kind_async(title, description, model=model))
