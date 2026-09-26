"""Judge how well a case's tests pin the behaviour its ticket describes.

`validate` already proves the tests fail before the fix and pass after it. That says the
tests are wired up, not that they are good: a test can fail for a trivial reason, assert
an implementation detail, or miss the behaviour the ticket actually asks for. This asks a
model to read the ticket and the test patch together and say so.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval_harness.collect.cases import Case, load_case
from eval_harness.config import ModelConfig
from eval_harness.metrics.judge import extract_json, run_sync
from eval_harness.paths import reviews_dir

MAX_PATCH_CHARS = 24_000

SYSTEM = (
    "You review test suites for an evaluation benchmark. You are given a ticket and the "
    "tests a developer added or changed when they fixed it. Judge only whether those tests "
    "would catch a wrong or missing implementation of what the ticket describes. "
    "Reply with a single JSON object and no prose."
)

PROMPT = """\
# Ticket

{ticket}

# Tests the developer added or changed

```diff
{test_patch}
```

# Files the fix itself touched (for context; you are not reviewing the fix)

{code_files}

Score this test suite as a benchmark case. Judge these separately:

- discriminating: would these tests fail for an implementation that misses the ticket's
  point, rather than only for a crash or a typo?
- covers_ticket: do they exercise the behaviour the ticket asks for, including the edge
  cases it names?
- specific: do they assert on behaviour rather than on incidental implementation detail
  that a correct alternative implementation would break?
- independent: can they pass without the tests themselves containing the answer, and
  without depending on unrelated state?

Reply with JSON with exactly these keys:
{{
  "discriminating": 0.0-1.0,
  "covers_ticket": 0.0-1.0,
  "specific": 0.0-1.0,
  "independent": 0.0-1.0,
  "score": 0.0-1.0,          // your overall judgement, not necessarily the mean
  "verdict": "strong" | "usable" | "weak",
  "reason": "two or three sentences, concrete about this case",
  "risks": ["short phrases naming anything that would make this a poor benchmark case"]
}}
"""


class TestQualityReview(BaseModel):
    case_id: str
    judge: str
    reviewed_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    discriminating: float | None = None
    covers_ticket: float | None = None
    specific: float | None = None
    independent: float | None = None
    score: float | None = None
    verdict: str = "unknown"
    reason: str = ""
    risks: list[str] = Field(default_factory=list)
    error: str | None = None


def review_path(case_id: str) -> Path:
    return reviews_dir() / f"{case_id}.json"


def load_review(case_id: str) -> TestQualityReview | None:
    path = review_path(case_id)
    if not path.exists():
        return None
    try:
        return TestQualityReview.model_validate_json(path.read_text())
    except Exception:
        return None


def save_review(review: TestQualityReview) -> Path:
    reviews_dir().mkdir(parents=True, exist_ok=True)
    path = review_path(review.case_id)
    path.write_text(json.dumps(review.model_dump(), indent=2) + "\n")
    return path


def build_prompt(case: Case) -> str:
    patch = case.human_test_patch
    if len(patch) > MAX_PATCH_CHARS:
        patch = patch[:MAX_PATCH_CHARS] + "\n… truncated …"
    return PROMPT.format(
        ticket=case.task_prompt,
        test_patch=patch,
        code_files="\n".join(f"- {f}" for f in case.code_files) or "- (none recorded)",
    )


def _coerce(data: dict[str, Any], case_id: str, judge_name: str) -> TestQualityReview:
    def num(key: str) -> float | None:
        v = data.get(key)
        try:
            return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    verdict = str(data.get("verdict") or "unknown").lower()
    if verdict not in ("strong", "usable", "weak"):
        verdict = "unknown"
    risks = data.get("risks") or []
    return TestQualityReview(
        case_id=case_id,
        judge=judge_name,
        discriminating=num("discriminating"),
        covers_ticket=num("covers_ticket"),
        specific=num("specific"),
        independent=num("independent"),
        score=num("score"),
        verdict=verdict,
        reason=str(data.get("reason") or ""),
        risks=[str(r) for r in risks][:8] if isinstance(risks, list) else [],
    )


def review_case(
    case_id: str, judge_cfg: ModelConfig, generate_fn: Any | None = None
) -> TestQualityReview:
    """Ask the judge about one case's tests and write the review to data/case-reviews/."""
    from eval_harness.adapters.oneshot import oneshot

    case = load_case(case_id)
    ask = generate_fn or oneshot
    judge_name = f"{judge_cfg.model} ({judge_cfg.provider}, effort={judge_cfg.effort})"
    try:
        text = run_sync(
            ask(
                build_prompt(case),
                system=SYSTEM,
                model=judge_cfg,
                max_tokens=judge_cfg.max_output_tokens,
            )
        )
        review = _coerce(extract_json(str(text)), case_id, judge_name)
    except Exception as e:
        review = TestQualityReview(
            case_id=case_id, judge=judge_name, error=f"{type(e).__name__}: {e}"[:500]
        )
    save_review(review)
    return review
