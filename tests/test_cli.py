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


def test_run_says_which_split_it_picked_and_how_many_exist(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The default split is the holdout; a whole benchmark ran on 9 of 31 cases without a word."""
    from eval_harness import cli

    fake = {"holdout": ["H1", "H2"], "all": ["H1", "H2", "D1", "D2", "D3"]}
    monkeypatch.setattr(cli, "_cases_for_split", lambda s: [object()] * len(fake[s]))

    def stop(*a, **k):  # type: ignore[no-untyped-def]
        raise SystemExit(0)

    monkeypatch.setattr("eval_harness.config.load_repos", stop)
    result = CliRunner().invoke(app, ["run", "--model", "noop"])
    assert (
        "running 2 case(s) from split 'holdout' of 5; --split all runs every case" in result.output
    )
