from typer.testing import CliRunner

from eval_harness.cli import app


def test_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "collect" in result.output
