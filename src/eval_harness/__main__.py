"""`python -m eval_harness` runs the CLI, so subprocesses need no console script on PATH."""

from eval_harness.cli import app

app()
