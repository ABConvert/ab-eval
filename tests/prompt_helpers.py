"""Shared set-up for the prompt-task tests: a data root with the fixture task imported."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from eval_harness.adapters.base import Usage
from eval_harness.adapters.oneshot import OneshotResult
from eval_harness.config import ModelConfig

FIXTURES = Path(__file__).parent / "fixtures" / "prompt-tasks"
TASK_DIR = FIXTURES / "demo-summary"
FAKE_SCORER = FIXTURES / "fake_scorer.py"
TASK = "demo-summary"

MODELS: dict[str, dict[str, Any]] = {
    "cand": {"provider": "anthropic", "model": "cand-1", "max_output_tokens": 4000},
    "judge-a": {"provider": "anthropic", "model": "judge-a-1", "max_output_tokens": 1000},
    "judge-b": {"provider": "anthropic", "model": "judge-b-1", "max_output_tokens": 1000},
    # Same model id as the candidate under another key: a judge rating itself.
    "judge-self": {"provider": "anthropic", "model": "cand-1", "max_output_tokens": 1000},
}


def clear_spend_env(monkeypatch: Any) -> None:
    """Unset both spend variables and have them restored afterwards.

    ensure_monthly_ledger writes os.environ itself; set-then-delete makes monkeypatch record
    the variable, so whatever the code under test sets is undone at teardown.
    """
    for name in ("ABEVAL_SPEND_LEDGER", "ABEVAL_SPEND_LIMIT_USD"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def model(key: str) -> ModelConfig:
    return ModelConfig.model_validate({"key": key, **MODELS[key]})


def setup_root(
    root: Path,
    *,
    scorer: bool = True,
    judges: list[str] | None = None,
    timeout: int = 20,
) -> None:
    """Import the fixture task into `root`, with the fake scorer and these judges."""
    from eval_harness import paths
    from eval_harness.prompt.cases import import_prompt_cases

    src = root / "_export"
    shutil.copytree(TASK_DIR, src)
    cfg = yaml.safe_load((src / "task.yaml").read_text())
    cfg["judges"] = judges if judges is not None else ["judge-a", "judge-b"]
    if scorer:
        cfg["scorer"] = {
            "command": [sys.executable, str(FAKE_SCORER)],
            "timeout_seconds": timeout,
        }
    (src / "task.yaml").write_text(yaml.safe_dump(cfg))
    config = root / "config"
    config.mkdir(exist_ok=True)
    (config / "models.yaml").write_text(yaml.safe_dump(MODELS))
    paths.reset_cache()
    import_prompt_cases(TASK, src)


def result(text: str, *, out: int = 100, cost: float = 0.01) -> OneshotResult:
    return OneshotResult(
        text=text,
        usage=Usage(input_tokens=50, output_tokens=out),
        cost_usd=cost,
        latency_seconds=0.5,
        request_settings={"provider": "fake"},
        stop_reason="end_turn",
    )
