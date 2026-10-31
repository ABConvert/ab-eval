"""The scorer protocol: stdin {case, output_text}, stdout {metrics, flags}, else an error."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eval_harness import paths
from eval_harness.prompt.cases import load_prompt_case
from eval_harness.prompt.scorer import ScorerError, parse_scorer_output, run_scorer
from eval_harness.prompt.task import ScorerConfig
from tests.prompt_helpers import FAKE_SCORER, TASK, clear_spend_env, setup_root


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    clear_spend_env(monkeypatch)
    paths.reset_cache()
    setup_root(tmp_path)
    return tmp_path


def test_scorer_output_must_be_numbers_or_booleans() -> None:
    ok = parse_scorer_output('{"metrics": {"a": 1, "b": true, "c": 0.5}, "flags": ["x"]}')
    assert ok.metrics == {"a": 1.0, "b": True, "c": 0.5} and ok.flags == ["x"]
    assert isinstance(ok.metrics["b"], bool), "a bool must stay a bool, not become 1.0"
    for bad in ('{"metrics": {"a": "0.8"}}', '{"metrics": {"a": NaN}}', "[1]", "nope"):
        with pytest.raises(ScorerError):
            parse_scorer_output(bad)
    with pytest.raises(ScorerError, match="flags"):
        parse_scorer_output('{"metrics": {}, "flags": "x"}')


def _scorer(timeout: int = 20) -> ScorerConfig:
    return ScorerConfig(command=[sys.executable, str(FAKE_SCORER)], timeout_seconds=timeout)


def test_the_fake_scorer_speaks_the_protocol(root: Path) -> None:
    case = load_prompt_case(TASK, "demo-sum-0001")
    out = run_scorer(_scorer(), case=case, output_text="A tidy interval run.", cwd=root)
    assert out.metrics["names_type"] is True and out.metrics["word_count"] == 4.0
    assert out.flags == ["too_short"]


@pytest.mark.parametrize(
    ("text", "match"),
    [("SCORER_CRASH", "exited 3"), ("SCORER_GARBAGE", "not JSON"), ("SCORER_SLEEP", "within 1s")],
)
def test_every_scorer_failure_is_an_error_not_a_score(root: Path, text: str, match: str) -> None:
    case = load_prompt_case(TASK, "demo-sum-0001")
    with pytest.raises(ScorerError, match=match):
        run_scorer(_scorer(timeout=1), case=case, output_text=text, cwd=root)


def test_a_missing_scorer_program_is_an_error(root: Path) -> None:
    case = load_prompt_case(TASK, "demo-sum-0001")
    cfg = ScorerConfig(command=["/no/such/scorer"])
    with pytest.raises(ScorerError, match="could not be started"):
        run_scorer(cfg, case=case, output_text="x", cwd=root)
