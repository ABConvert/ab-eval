from typer.testing import CliRunner

from eval_harness.cli import app


def test_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "collect" in result.output


def test_curate_lists_then_records_the_pending_list(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from eval_harness import paths
    from eval_harness.collect.curation import Candidate, load_verdicts, save_pending

    monkeypatch.setenv("ABEVAL_DATA_ROOT", str(tmp_path))
    paths.reset_cache()
    save_pending(
        "demo-app",
        [
            Candidate(key="DEMO-5", pr=5, title="fix the thing", author="a", files=2, lines=40),
            Candidate(key="DEMO-6", pr=6, title="add a thing", author="b", files=3, lines=90),
        ],
    )
    listed = CliRunner().invoke(app, ["curate", "--repo", "demo-app"])
    assert listed.exit_code == 0 and "DEMO-5" in listed.output and "2 of 2" in listed.output

    done = CliRunner().invoke(
        app,
        [
            "curate",
            "--repo",
            "demo-app",
            "--accept",
            "demo-5",
            "--kind",
            "bug_fix",
            "--reject",
            "DEMO-6",
        ],
    )
    assert done.exit_code == 0, done.output
    verdicts = load_verdicts()
    assert verdicts["DEMO-5"].tier == "A" and verdicts["DEMO-5"].kind_review == "bug_fix"
    assert verdicts["DEMO-6"].tier == "X"

    missing = CliRunner().invoke(app, ["curate", "--repo", "demo-app", "--accept", "NOPE-1"])
    assert missing.exit_code == 1 and "NOPE-1" in missing.output
