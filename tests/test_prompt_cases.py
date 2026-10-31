"""Prompt cases, task.yaml/rubric.yaml, `import prompt-cases`, and datasets over them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from eval_harness import paths
from eval_harness.prompt.cases import (
    ImportRejected,
    PromptCase,
    import_prompt_cases,
    load_prompt_case,
    prompt_case_ids,
)
from eval_harness.prompt.task import Dimension, Rubric, TaskConfig, load_task

FIXTURE = Path(__file__).parent / "fixtures" / "prompt-tasks" / "demo-summary"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    return tmp_path


def _export(tmp_path: Path) -> Path:
    """A copy of the fixture export, to break in one test without touching the others."""
    out = tmp_path / "export"
    shutil.copytree(FIXTURE, out)
    return out


# ------------------------------------------------------------------ the case model


def test_the_fixture_cases_are_valid_prompt_cases() -> None:
    for path in FIXTURE.glob("*.json"):
        case = PromptCase.model_validate_json(path.read_text())
        assert case.kind == "prompt" and case.task == "demo-summary"
        assert case.grading_context, "the fixture exists to exercise grading_context"


def test_a_misspelt_field_is_refused_not_dropped() -> None:
    """`grading_contxt` silently ignored would hand every judge an empty context."""
    body = json.loads((FIXTURE / "demo-sum-0001.json").read_text())
    body["grading_contxt"] = body.pop("grading_context")
    with pytest.raises(ValidationError, match="grading_contxt"):
        PromptCase.model_validate(body)


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", ".hidden", "x" * 200])
def test_case_ids_must_be_safe_file_names(bad: str) -> None:
    body = json.loads((FIXTURE / "demo-sum-0001.json").read_text())
    body["case_id"] = bad
    with pytest.raises(ValidationError):
        PromptCase.model_validate(body)


# ------------------------------------------------------------------ task.yaml and rubric


def test_fixture_task_and_rubric_load(root: Path) -> None:
    import_prompt_cases("demo-summary", FIXTURE)
    cfg = load_task("demo-summary")
    rubric = cfg.load_rubric()
    assert [d.name for d in rubric.dimensions] == ["faithfulness", "usefulness", "type_correct"]
    assert rubric.dimension("type_correct").is_enum
    assert rubric.dimension("faithfulness").bounds == (1, 5)
    assert cfg.scorer is None, "a scorer is optional"


def test_a_task_with_only_a_name_gets_todays_defaults() -> None:
    cfg = TaskConfig(task="t")
    assert cfg.rubric == "rubric.yaml" and cfg.judges == [] and cfg.defaults.repeats == 1
    assert cfg.defaults.max_output_tokens is None


def test_rubric_hash_moves_when_an_instruction_does() -> None:
    a = Rubric(dimensions=[Dimension(name="x", scale="1-5", instruction="be strict")])
    b = Rubric(dimensions=[Dimension(name="x", scale="1-5", instruction="be lenient")])
    assert a.hash() != b.hash()
    assert a.hash() == Rubric.model_validate(a.model_dump()).hash()


@pytest.mark.parametrize(
    "dim",
    [
        {"name": "x", "scale": "5-1", "instruction": "i"},
        {"name": "x", "scale": "1to5", "instruction": "i"},
        {"name": "x", "scale": ["only"], "instruction": "i"},
        {"name": "x", "scale": ["a", "b"], "good": ["c"], "instruction": "i"},
        {"name": "x", "scale": "1-5", "good": ["5"], "instruction": "i"},
        {"name": "x", "scale": "1-5", "instruction": "i", "uses": ["grading_context"]},
        {"name": "Bad Name", "scale": "1-5", "instruction": "i"},
    ],
)
def test_malformed_dimensions_are_refused(dim: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Dimension.model_validate(dim)


def test_dimension_coerce_takes_only_values_on_the_scale() -> None:
    scale = Dimension(name="x", scale="1-5", instruction="i")
    assert scale.coerce("4") == 4 and scale.coerce(5.0) == 5
    for bad in (0, 6, 3.5, "four", True):
        with pytest.raises(ValueError):
            scale.coerce(bad)
    enum = Dimension(name="y", scale=["yes", "no"], instruction="i")
    assert enum.coerce(" yes ") == "yes"
    with pytest.raises(ValueError):
        enum.coerce("maybe")


def test_rubric_must_sit_beside_task_yaml() -> None:
    with pytest.raises(ValidationError):
        TaskConfig(task="t", rubric="../elsewhere/rubric.yaml")


# ------------------------------------------------------------------ import


def test_import_copies_cases_and_config(root: Path) -> None:
    rep = import_prompt_cases("demo-summary", FIXTURE)
    assert sorted(rep.added) == ["demo-sum-0001", "demo-sum-0002", "demo-sum-0003"]
    assert set(rep.config_copied) == {"task.yaml", "rubric.yaml"}
    assert prompt_case_ids("demo-summary") == sorted(rep.added)
    again = import_prompt_cases("demo-summary", FIXTURE)
    assert again.added == [] and len(again.unchanged) == 3


def test_a_changed_prompt_under_an_old_id_is_reported(root: Path, tmp_path: Path) -> None:
    import_prompt_cases("demo-summary", FIXTURE)
    src = _export(tmp_path)
    path = src / "demo-sum-0001.json"
    body = json.loads(path.read_text())
    body["prompt"] += " Be brief."
    path.write_text(json.dumps(body))
    assert import_prompt_cases("demo-summary", src).updated == ["demo-sum-0001"]


def test_one_bad_file_rejects_the_whole_import(root: Path, tmp_path: Path) -> None:
    """A dataset silently missing what the exporter could not serialise is a biased one."""
    src = _export(tmp_path)
    (src / "demo-sum-0002.json").write_text('{"case_id": "demo-sum-0002"}')
    with pytest.raises(ImportRejected) as caught:
        import_prompt_cases("demo-summary", src)
    assert any("demo-sum-0002" in p for p in caught.value.problems)
    assert prompt_case_ids("demo-summary") == []


def test_import_refuses_cases_of_another_task(root: Path) -> None:
    with pytest.raises(ImportRejected, match="not 'other-task'"):
        import_prompt_cases("other-task", FIXTURE)


def test_import_refuses_a_file_named_unlike_its_case(root: Path, tmp_path: Path) -> None:
    src = _export(tmp_path)
    (src / "demo-sum-0001.json").rename(src / "renamed.json")
    with pytest.raises(ImportRejected, match="does not match case_id"):
        import_prompt_cases("demo-summary", src)


def test_import_needs_a_task_yaml_somewhere(root: Path, tmp_path: Path) -> None:
    src = _export(tmp_path)
    (src / "task.yaml").unlink()
    with pytest.raises(ImportRejected, match=r"no task\.yaml"):
        import_prompt_cases("demo-summary", src)


def test_holdout_fraction_splits_only_unsplit_cases_by_date(root: Path, tmp_path: Path) -> None:
    src = _export(tmp_path)
    for path in src.glob("*.json"):
        body = json.loads(path.read_text())
        body["split"] = None
        path.write_text(json.dumps(body))
    rep = import_prompt_cases("demo-summary", src, holdout_fraction=0.34)
    assert rep.splits_assigned == 3
    splits = {i: load_prompt_case("demo-summary", i).split for i in prompt_case_ids("demo-summary")}
    assert splits["demo-sum-0003"] == "holdout", "the newest case is the held-out one"
    assert splits["demo-sum-0001"] == "dev"


def test_cli_import_prompt_cases(root: Path) -> None:
    from eval_harness.cli import app

    out = CliRunner().invoke(
        app, ["import", "prompt-cases", "--task", "demo-summary", "--from", str(FIXTURE)]
    )
    assert out.exit_code == 0, out.output
    assert "3 added" in out.output


def test_cli_import_reports_every_problem(root: Path, tmp_path: Path) -> None:
    from eval_harness.cli import app

    src = _export(tmp_path)
    (src / "demo-sum-0001.json").write_text("{}")
    out = CliRunner().invoke(
        app, ["import", "prompt-cases", "--task", "demo-summary", "--from", str(src)]
    )
    assert out.exit_code == 1
    assert "nothing imported" in out.output and "demo-sum-0001" in out.output


# ------------------------------------------------------------------ datasets


def test_builtin_datasets_over_a_task_follow_each_cases_split(root: Path) -> None:
    from eval_harness.collect import datasets

    import_prompt_cases("demo-summary", FIXTURE)
    assert datasets.load("all", "demo-summary").case_ids == [
        "demo-sum-0001",
        "demo-sum-0002",
        "demo-sum-0003",
    ]
    assert datasets.load("holdout", "demo-summary").case_ids == ["demo-sum-0003"]
    assert datasets.load("dev", "demo-summary").case_ids == ["demo-sum-0001", "demo-sum-0002"]
    # Without a task the built-ins still mean coding cases, as before.
    assert datasets.load("all").case_ids == []


def test_a_saved_dataset_can_hold_prompt_cases(root: Path) -> None:
    from eval_harness.collect import datasets

    import_prompt_cases("demo-summary", FIXTURE)
    datasets.create("pilot", "two prompt cases")
    ds = datasets.add_cases("pilot", ["demo-sum-0002", "demo-sum-0001"])
    assert ds.case_ids == ["demo-sum-0002", "demo-sum-0001"]
    with pytest.raises(LookupError):
        datasets.add_cases("pilot", ["no-such-case"])
    assert datasets.matching(["demo-sum-0001", "demo-sum-0002"], "demo-summary") == "pilot"
    assert datasets.matching(["demo-sum-0003"], "demo-summary") == "holdout"
