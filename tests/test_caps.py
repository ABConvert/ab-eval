"""Where a run's budget comes from, and whether the artifact says what it was.

The bench-v2 runs shared one wall clock because there was nowhere else to put one. It
stopped the slowest model twenty-seven times and the fastest none, and `summary.json` did
not record it, so the published numbers could not be reproduced from their own artifact.
These tests hold the three layers apart — harness default, the model's own `caps:` block,
a flag passed for one run — and hold the summary to recording what came out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.text import Text
from typer.testing import CliRunner

from eval_harness.adapters.base import Caps
from eval_harness.config import ModelConfig, load_models
from tests.conftest import FIXTURES


def _models(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(body)
    return path


# --------------------------------------------------------------- the caps block in config


def test_a_caps_block_in_models_yaml_is_honoured(tmp_path: Path) -> None:
    """The point of the block: a slow local model carries its own clock, in config."""
    path = _models(
        tmp_path,
        "slow-local:\n"
        "  provider: openai-compat\n"
        "  model: qwen\n"
        "  base_url: http://gpu-box:11434/v1\n"
        "  api_key_env: OLLAMA_API_KEY\n"
        "  caps:\n"
        "    wall_clock_seconds: 5400\n"
        "    max_turns: 60\n",
    )
    cfg = load_models(path)["slow-local"]
    assert cfg.caps is not None
    caps = Caps.resolve(cfg.caps.model_dump())
    assert caps.wall_clock_seconds == 5400
    assert caps.max_turns == 60
    # What the block does not name keeps the harness default, so adding a longer clock does
    # not silently change the token budget with it.
    assert caps.max_output_tokens_total == 120_000
    assert caps.tool_timeout_seconds == 300


def test_a_model_with_no_caps_block_gets_exactly_the_old_defaults(tmp_path: Path) -> None:
    path = _models(tmp_path, "plain:\n  provider: anthropic\n  model: m\n")
    cfg = load_models(path)["plain"]
    assert cfg.caps is None
    assert Caps.resolve(None) == Caps()


@pytest.mark.parametrize("block", ["caps: {}", "caps:"])
def test_an_empty_or_null_caps_block_overrides_nothing(tmp_path: Path, block: str) -> None:
    path = _models(tmp_path, f"plain:\n  provider: anthropic\n  model: m\n  {block}\n")
    cfg = load_models(path)["plain"]
    declared = cfg.caps.model_dump() if cfg.caps else None
    assert Caps.resolve(declared) == Caps()


def test_a_misspelled_cap_is_refused_rather_than_dropped(tmp_path: Path) -> None:
    """`wallclock_seconds` accepted and ignored would be this bug in a new place."""
    path = _models(
        tmp_path,
        "typo:\n  provider: anthropic\n  model: m\n  caps:\n    wallclock_seconds: 5400\n",
    )
    with pytest.raises(Exception, match="wallclock_seconds"):
        load_models(path)


# ------------------------------------------------------------------ flag beats block, once


def test_an_explicit_flag_beats_the_models_caps_block() -> None:
    caps = Caps.resolve({"wall_clock_seconds": 5400, "max_turns": 60}, max_turns=12)
    assert caps.max_turns == 12
    # Overriding one cap leaves the others where the model put them.
    assert caps.wall_clock_seconds == 5400


def test_an_unpassed_flag_does_not_beat_the_models_caps_block() -> None:
    """The whole discipline: an unpassed flag is None, never the number it defaults to.

    A typer option declared `int = 40` cannot tell "not passed" from "someone typed 40", so
    config would lose every single run while appearing to work.
    """
    caps = Caps.resolve(
        {"wall_clock_seconds": 5400, "max_turns": 60},
        max_turns=None,
        wall_clock_seconds=None,
        max_output_tokens_total=None,
        tool_timeout_seconds=None,
    )
    assert caps.max_turns == 60
    assert caps.wall_clock_seconds == 5400


@pytest.mark.parametrize("color", [False, True])
def test_the_run_command_exposes_a_flag_for_every_cap(color: bool) -> None:
    """Two of the four had no flag at all; `Caps()` defaults were the only way to set them."""
    from eval_harness.cli import app

    result = CliRunner().invoke(app, ["run", "--help"], color=color, terminal_width=120)
    assert result.exit_code == 0
    # Rich can color parts of an option independently on CI's terminal. Compare the
    # displayed text, not ANSI bytes inserted between the dashes and option name.
    flat = " ".join(Text.from_ansi(result.output).plain.split())
    for flag in ("--max-turns", "--wall-clock", "--max-output-tokens", "--tool-timeout"):
        assert flag in flat, flag
    # And none of them carries a default, which is what would make it always win.
    assert "[default: 40]" not in flat and "[default: 1800]" not in flat


def _data_root(root: Path, models_yaml: str) -> str:
    """A relocated data root with the fixture repo, one case, and the given models.yaml.

    Returns the case id. Cases carry the fixture repository's github slug because `run`
    resolves the repo by matching it.
    """
    import shutil

    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "config" / "repos.yaml", root / "config" / "repos.yaml")
    (root / "config" / "models.yaml").write_text(models_yaml)
    cases = root / "data" / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    (cases / "DEMO-1.json").write_text(json.dumps(_case("DEMO-1")))
    return "DEMO-1"


def _case(case_id: str) -> dict[str, Any]:
    """A minimal valid case on the fixture repository, whose slug `run` matches on."""
    return {
        "case_id": case_id,
        "repo": "acme/demo-app",
        "kind": "bug_fix",
        "base_commit": "c" * 40,
        "task_prompt": "do the thing",
        "test_command": "x",
        "test_files_added": [],
        "test_files": ["a.test.ts"],
        "code_files": ["a.ts"],
        "human_patch": "",
        "human_test_patch": "",
        "files_changed": 3,
        "lines_changed": 100,
        "human_cycle_time_hours": 1.0,
        "review_comment_count": 1,
        "merged_at": "",
        "pr_number": 1,
        "merge_commit": "m",
        "test_runner": "unit",
        "files_dropped": [],
        "tier": "A",
        "kind_source": "review",
        "author": "a",
        "author_kind": "human",
        "spec_level_heuristic": "solution-specified",
        "ticket_created_at": "",
        "ticket_started_at": None,
        "human_cycle_time_source": "created_at",
    }


def test_the_run_command_resolves_caps_from_config_and_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through `eval-harness run`: what lands in run.json, without a container.

    The run is stopped at `run_many`, the last thing before Docker; everything under test —
    loading the model, resolving the caps, writing run.json — has already happened.
    """
    from eval_harness import paths
    from eval_harness.cli import app

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    case = _data_root(
        tmp_path,
        "slow-local:\n"
        "  provider: openai-compat\n"
        "  model: qwen\n"
        "  base_url: http://gpu-box:11434/v1\n"
        "  api_key_env: OLLAMA_API_KEY\n"
        "  caps:\n    wall_clock_seconds: 5400\n    tool_timeout_seconds: 900\n",
    )
    monkeypatch.setenv("OLLAMA_API_KEY", "x")

    async def _stop(*a: Any, **kw: Any) -> list[Any]:
        return []

    monkeypatch.setattr("eval_harness.harness.runner.run_many", _stop)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--model",
            "slow-local",
            "--case",
            case,
            "--run-id",
            "t1",
            "--max-turns",
            "7",
            "--no-score",
        ],
    )
    assert result.exit_code == 0, result.output
    meta = json.loads((tmp_path / "results" / "t1" / "run.json").read_text())
    assert meta["caps"] == {
        "max_turns": 7,  # the flag that was passed
        "wall_clock_seconds": 5400,  # the model's own, not lost to an unpassed flag
        "tool_timeout_seconds": 900,  # likewise, and it had no flag at all before this
        "max_output_tokens_total": 120_000,  # named nowhere, so the harness default
    }


# ------------------------------------------------------------- the summary records the caps


def test_the_effective_caps_reach_summary_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published result has to say what budget produced it."""
    from eval_harness import paths
    from eval_harness.report.summary import RunSummary, load_summary, save_summary

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    caps = {
        "max_turns": 60,
        "max_output_tokens_total": 120_000,
        "wall_clock_seconds": 5400,
        "tool_timeout_seconds": 900,
    }
    save_summary(
        RunSummary(
            run_id="r1",
            model="slow-local",
            model_config_snapshot={},
            caps=caps,
            harness_git_sha="abc",
            judge=None,
            judge_config=None,
            scored_at="2026-01-01T00:00:00Z",
            cases=[],
        )
    )
    on_disk = json.loads((tmp_path / "results" / "r1" / "summary.json").read_text())
    assert on_disk["caps"] == caps
    assert load_summary("r1").caps == caps


def test_score_run_copies_the_caps_the_run_actually_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run.json has carried the caps all along; this is the wire from there to the summary."""
    from eval_harness import paths
    from eval_harness.metrics import dataset

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    _data_root(tmp_path, "judge:\n  provider: anthropic\n  model: m\n")
    run_dir = tmp_path / "results" / "r2"
    run_dir.mkdir(parents=True)
    caps = {
        "max_turns": 60,
        "max_output_tokens_total": 120_000,
        "wall_clock_seconds": 5400,
        "tool_timeout_seconds": 300,
    }
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "r2", "model": "slow-local", "caps": caps, "harness_git_sha": "abc"})
    )
    (run_dir / "DEMO-1.json").write_text(
        json.dumps(
            {
                "case_id": "DEMO-1",
                "run_id": "r2",
                "model": "slow-local",
                "status": "completed",
                "cap_hit": "wall_clock",
            }
        )
    )
    summary = dataset.score_run("r2", judge_cfg=None, use_cache=False)
    assert summary.caps == caps


def test_a_summary_written_before_this_field_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All eight bench-v2 summaries predate `caps`, and they have to keep opening."""
    from eval_harness import paths
    from eval_harness.report.summary import load_summary

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    run_dir = tmp_path / "results" / "old"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "old",
                "model": "m",
                "model_config_snapshot": {},
                "harness_git_sha": "abc",
                "judge": None,
                "judge_config": None,
                "scored_at": "2026-01-01T00:00:00Z",
                "cases": [],
                "metrics": {},
            }
        )
    )
    assert load_summary("old").caps == {}


def test_every_recorded_bench_v2_summary_still_loads() -> None:
    """Not a synthetic old summary: the eight results the numbers were published from."""
    from eval_harness import PROJECT_ROOT
    from eval_harness.report.summary import RunSummary

    found = sorted((PROJECT_ROOT / "results").glob("bench2-*/summary.json"))
    if not found:
        pytest.skip("this checkout has no bench-v2 results")
    for path in found:
        assert RunSummary.model_validate_json(path.read_text()).run_id


def test_the_committed_models_file_still_loads() -> None:
    """New fields are optional or this breaks, and it is the file the operator runs on.

    Only that it loads. Asserting the new fields are unset here would be a test that fails
    the moment someone gives qwen3.8 the longer clock this work exists to let them give it.
    `tests/fixtures/config/models.yaml` is the fixed file the next test holds to the old
    behaviour, because it is the one nobody is meant to edit.
    """
    from eval_harness import PROJECT_ROOT

    path = PROJECT_ROOT / "config" / "models.yaml"
    if not path.exists():
        pytest.skip("this checkout has no models.yaml")
    assert load_models(path)


def test_a_config_predating_every_new_field_behaves_as_it_did() -> None:
    """The fixture config names none of them, so each one has to resolve to the old value."""
    models = load_models(FIXTURES / "config" / "models.yaml")
    assert models
    for cfg in models.values():
        assert cfg.caps is None
        assert Caps.resolve(None) == Caps()
        assert cfg.temperature is None, "nothing is sent, as before"
        assert cfg.reasoning_param == "omit", "no reasoning field is sent, as before"


# ------------------------------------------------- the dashboard does not override in secret


def test_a_blank_cap_on_the_launch_form_passes_no_flag() -> None:
    """The form used to post 40 and 1800 every time, so a caps block never applied to it."""
    from eval_harness.dashboard import jobs

    job = jobs.run_job(model="m", run_id="r", cases=["c1"], max_turns=None, wall_clock=None)
    assert "--max-turns" not in job.argv
    assert "--wall-clock" not in job.argv


def test_a_cap_typed_on_the_launch_form_is_passed() -> None:
    from eval_harness.dashboard import jobs

    job = jobs.run_job(model="m", run_id="r", cases=["c1"], max_turns=12, wall_clock=None)
    assert job.argv[job.argv.index("--max-turns") + 1] == "12"
    assert "--wall-clock" not in job.argv


# ----------------------------------------------------------------- the form writes the block


def test_the_setup_form_writes_and_reads_back_a_caps_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from starlette.testclient import TestClient

    from eval_harness import paths
    from eval_harness.dashboard.app import build_app

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "models.yaml").write_text(
        "judge:\n  provider: anthropic\n  model: m\n  effort: medium\n"
    )
    client = TestClient(build_app())
    reply = client.post(
        "/setup/models",
        data={
            "key": "slow-local",
            "provider": "openai-compat",
            "model": "qwen",
            "effort": "high",
            "max_output_tokens": "16000",
            "base_url": "http://gpu-box:11434/v1",
            "api_key_env": "OLLAMA_API_KEY",
            "caps_wall_clock_seconds": "5400",
            "caps_max_turns": "60",
            "caps_max_output_tokens_total": "",
            "caps_tool_timeout_seconds": "",
        },
        headers={"sec-fetch-site": "same-origin"},
        follow_redirects=False,
    )
    assert reply.status_code == 303, reply.text
    written = yaml.safe_load((tmp_path / "config" / "models.yaml").read_text())
    assert written["slow-local"]["caps"] == {"wall_clock_seconds": 5400, "max_turns": 60}
    # And the harness's own loader accepts what the form wrote.
    assert load_models(tmp_path / "config" / "models.yaml")["slow-local"].caps is not None


def test_the_setup_form_leaves_out_caps_when_every_box_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from starlette.testclient import TestClient

    from eval_harness import paths
    from eval_harness.dashboard.app import build_app

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "models.yaml").write_text(
        "judge:\n  provider: anthropic\n  model: m\n  effort: medium\n"
    )
    client = TestClient(build_app())
    client.post(
        "/setup/models",
        data={
            "key": "plain",
            "provider": "anthropic",
            "model": "m",
            "effort": "high",
            "max_output_tokens": "16000",
            "caps_max_turns": "",
            "caps_wall_clock_seconds": "",
        },
        headers={"sec-fetch-site": "same-origin"},
    )
    written = yaml.safe_load((tmp_path / "config" / "models.yaml").read_text())
    assert "caps" not in written["plain"]


def test_the_setup_form_refuses_a_fractional_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from starlette.testclient import TestClient

    from eval_harness import paths
    from eval_harness.dashboard.app import build_app

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    (tmp_path / "config").mkdir(parents=True)
    before = "judge:\n  provider: anthropic\n  model: m\n  effort: medium\n"
    (tmp_path / "config" / "models.yaml").write_text(before)
    client = TestClient(build_app())
    reply = client.post(
        "/setup/models",
        data={
            "key": "plain",
            "provider": "anthropic",
            "model": "m",
            "effort": "high",
            "max_output_tokens": "16000",
            "caps_max_turns": "7.5",
        },
        headers={"sec-fetch-site": "same-origin"},
    )
    assert reply.status_code == 422
    assert "caps.max_turns" in reply.text
    assert (tmp_path / "config" / "models.yaml").read_text() == before, "nothing was written"


def test_model_config_round_trips_through_its_own_snapshot() -> None:
    """`run.json` stores `model_dump()`; the new fields have to survive that."""
    cfg = ModelConfig(
        key="slow-local",
        provider="openai-compat",
        model="qwen",
        temperature=0.2,
        reasoning_param="reasoning_effort",
        caps={"wall_clock_seconds": 5400},  # type: ignore[arg-type]
    )
    again = ModelConfig.model_validate(cfg.model_dump())
    assert again == cfg
    assert again.caps is not None and again.caps.wall_clock_seconds == 5400
