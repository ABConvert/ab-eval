"""Which judge scores a run, and where a second judge's opinion goes.

A judge that is one of the models on the leaderboard grades its own family. Swapping it
means naming a different entry in models.yaml; checking one judge against another means
scoring the same run twice without the second pass overwriting the first. These tests hold
both: `--judge-key` picks the entry, `--write-as` keeps a second opinion beside the first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.test_caps import _data_root

TWO_JUDGES = (
    "judge:\n  provider: anthropic\n  model: first-judge\n"
    "judge-b:\n  provider: anthropic\n  model: second-judge\n"
)


def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models_yaml: str) -> None:
    from eval_harness import paths

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    _data_root(tmp_path, models_yaml)


def _capture_score_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake(run_id: str, **kw: Any) -> Any:
        seen.update(kw, run_id=run_id)

        from types import SimpleNamespace

        judge = kw["judge_cfg"].model if kw.get("judge_cfg") else None
        return SimpleNamespace(cases=[], judge=judge)

    monkeypatch.setattr("eval_harness.metrics.dataset.score_run", fake)
    return seen


def test_the_default_judge_is_still_the_entry_called_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval_harness.cli import app

    _root(tmp_path, monkeypatch, TWO_JUDGES)
    seen = _capture_score_run(monkeypatch)
    result = CliRunner().invoke(app, ["score", "--run", "r1"])
    assert result.exit_code == 0, result.output
    assert seen["judge_cfg"].model == "first-judge"
    assert seen["write_as"] is None


def test_judge_key_picks_a_different_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eval_harness.cli import app

    _root(tmp_path, monkeypatch, TWO_JUDGES)
    seen = _capture_score_run(monkeypatch)
    result = CliRunner().invoke(app, ["score", "--run", "r1", "--judge-key", "judge-b"])
    assert result.exit_code == 0, result.output
    assert seen["judge_cfg"].model == "second-judge"


def test_an_unknown_judge_key_scores_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo must not fall back to scoring with no judge and look like a finished pass."""
    from eval_harness.cli import app

    _root(tmp_path, monkeypatch, TWO_JUDGES)
    seen = _capture_score_run(monkeypatch)
    result = CliRunner().invoke(app, ["score", "--run", "r1", "--judge-key", "judge-typo"])
    assert result.exit_code != 0
    assert "judge-typo" in result.output
    assert seen == {}


def test_write_as_keeps_the_first_judges_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second opinion lands beside summary.json; the one the dashboard reads is untouched."""
    from eval_harness.metrics import dataset

    _root(tmp_path, monkeypatch, TWO_JUDGES)
    run_dir = tmp_path / "results" / "r2"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": "r2", "model": "m"}))
    (run_dir / "DEMO-1.json").write_text(
        json.dumps({"case_id": "DEMO-1", "run_id": "r2", "model": "m", "status": "completed"})
    )
    first = '{"sentinel": "the first judge wrote this"}\n'
    (run_dir / "summary.json").write_text(first)

    dataset.score_run("r2", judge_cfg=None, use_cache=False, write_as="summary.judge-b.json")

    assert (run_dir / "summary.json").read_text() == first
    second = json.loads((run_dir / "summary.judge-b.json").read_text())
    assert second["run_id"] == "r2"


@pytest.mark.parametrize("bad", ["../escape.json", "sub/summary.json", "summary.txt"])
def test_write_as_stays_a_json_file_in_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    from eval_harness.metrics import dataset

    _root(tmp_path, monkeypatch, TWO_JUDGES)
    run_dir = tmp_path / "results" / "r3"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": "r3", "model": "m"}))
    (run_dir / "DEMO-1.json").write_text(
        json.dumps({"case_id": "DEMO-1", "run_id": "r3", "model": "m", "status": "completed"})
    )
    with pytest.raises(ValueError):
        dataset.score_run("r3", judge_cfg=None, use_cache=False, write_as=bad)


def test_a_run_whose_judge_hit_the_spend_limit_writes_no_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DeepEval ignores metric errors, so without this a stopped judge looks like a finished one."""
    from eval_harness.adapters import oneshot
    from eval_harness.metrics import dataset

    _root(tmp_path, monkeypatch, TWO_JUDGES)
    run_dir = tmp_path / "results" / "r4"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": "r4", "model": "m"}))
    (run_dir / "DEMO-1.json").write_text(
        json.dumps({"case_id": "DEMO-1", "run_id": "r4", "model": "m", "status": "completed"})
    )
    ledger = tmp_path / "spend.jsonl"
    ledger.write_text(json.dumps({"cost_usd": 25.0}) + "\n")
    monkeypatch.setenv(oneshot.LEDGER_ENV, str(ledger))
    monkeypatch.setenv(oneshot.LIMIT_ENV, "20")
    with pytest.raises(oneshot.SpendLimitReached):
        dataset.score_run("r4", judge_cfg=None, use_cache=False, write_as="summary.g.json")
    assert not (run_dir / "summary.g.json").exists()


def test_a_second_judges_summary_is_not_read_back_as_an_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring a run twice must work: the first pass's file sits in the same directory."""
    from eval_harness.metrics import dataset

    _root(tmp_path, monkeypatch, TWO_JUDGES)
    run_dir = tmp_path / "results" / "r5"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": "r5", "model": "m"}))
    (run_dir / "DEMO-1.json").write_text(
        json.dumps({"case_id": "DEMO-1", "run_id": "r5", "model": "m", "status": "completed"})
    )
    (run_dir / "summary.fable-judge.json").write_text('{"run_id": "r5", "cases": []}')
    dataset.score_run("r5", judge_cfg=None, use_cache=False, write_as="summary.b.json")
    dataset.score_run("r5", judge_cfg=None, use_cache=False, write_as="summary.c.json")
    assert len(json.loads((run_dir / "summary.c.json").read_text())["cases"]) == 1


def test_write_as_must_look_like_a_summary() -> None:
    """Only summary.* is skipped when records are loaded, so only summary.* may be written."""
    from eval_harness.metrics import dataset

    with pytest.raises(ValueError):
        dataset.score_run("any", judge_cfg=None, write_as="gemini.json")
