"""The product-owned scorer: one subprocess per attempt, JSON in and JSON out.

stdin  {"case": <the prompt case, grading_context included>, "output_text": "..."}
stdout {"metrics": {"name": number | bool, ...}, "flags": ["...", ...]}

The scorer runs in the product's own environment (its `command` is typically `uv run
--project <product checkout> ...`), which is how a product's checks reuse product code
without ab-eval importing it. Anything other than exit 0 with that shape is a `ScorerError`,
and the attempt it was scoring is marked `score_error` — never scored as zero, because a
broken scorer and a bad output must not look alike.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from eval_harness.prompt.cases import PromptCase
from eval_harness.prompt.task import ScorerConfig


class ScorerError(RuntimeError):
    pass


class ScorerOutput(BaseModel):
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


def parse_scorer_output(stdout: str) -> ScorerOutput:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ScorerError(f"scorer printed something that is not JSON: {stdout[:200]!r}") from e
    if not isinstance(data, dict) or not isinstance(data.get("metrics", {}), dict):
        raise ScorerError(f"scorer output must be an object with a metrics object: {data!r}"[:300])
    metrics: dict[str, float | bool] = {}
    for name, value in (data.get("metrics") or {}).items():
        # Checked by hand rather than by pydantic, whose lax mode reads "0.8" as 0.8 and
        # would let a scorer that prints strings look like one that prints numbers.
        if isinstance(value, bool):
            metrics[str(name)] = value
        elif isinstance(value, int | float) and math.isfinite(value):
            metrics[str(name)] = float(value)
        else:
            raise ScorerError(f"metric {name!r} is {value!r}; metrics are numbers or booleans")
    flags = data.get("flags") or []
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise ScorerError(f"flags must be a list of strings: {flags!r}"[:300])
    return ScorerOutput(metrics=metrics, flags=flags)


def run_scorer(cfg: ScorerConfig, *, case: PromptCase, output_text: str, cwd: Path) -> ScorerOutput:
    payload = json.dumps({"case": case.model_dump(), "output_text": output_text})
    try:
        proc = subprocess.run(
            cfg.command,
            input=payload,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        raise ScorerError(f"scorer gave no answer within {cfg.timeout_seconds}s") from None
    except OSError as e:
        raise ScorerError(f"scorer could not be started: {e}") from e
    if proc.returncode != 0:
        raise ScorerError(f"scorer exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
    return parse_scorer_output(proc.stdout)
