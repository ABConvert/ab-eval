# Contributing

## Setup

```bash
uv sync
uv run pytest -q          # unit tests, no Docker needed
uv run ruff check .
uv run ruff format --check .
pre-commit install        # gitleaks + ruff before every commit
```

The Docker-backed integration tests are marked and skipped by default:

```bash
uv run pytest -m integration    # needs Docker and a configured repo
```

## What makes a change easy to accept

- **A test that fails without it.** For a bug, write the failing test first and say in the
  message what it caught. Several fixes in this repository exist because a test was written to
  prove the fix was needed — that is the standard.
- **Measured claims.** "Faster" wants a number. This is a measurement tool; it should hold
  itself to the same bar.
- **Say what you did not do.** A change that fixes one of three related problems is welcome if
  it says so.

## Adding a test runner

`runners[].kind` is a closed set — `vitest`, `pytest`, `flutter`. Adding one means:

1. A command builder and an output parser in `src/eval_harness/harness/testrun.py`.
2. A fixture under `tests/fixtures/` holding real output from that runner, both passing and
   failing, and a parser test against it.
3. The `Literal` in `src/eval_harness/config.py` extended.

The parser must report totals, failures and **individual failed test names** — named tests are
what lets a case say "these must pass" rather than "this file must be clean".

## Adding a model provider

A module under `src/eval_harness/adapters/` implementing the `ModelAdapter` protocol, plus a
branch in `registry.py`. If it should also be usable as the code-quality judge, add a branch to
`adapters/oneshot.py`.

Adapters must never log a credential. Read the key from the environment variable named in
`models.yaml`, hold it in the instance, and keep it out of records, traces and errors.

## Adding a ticket source

Implement `TicketSource` in `src/eval_harness/collect/tickets.py`. Whatever your tracker
returns, run it through `sanitize.py` before it reaches a case file.

## Style

`ruff` decides formatting; do not argue with it in review. Comments explain *why* — the code
already says what. Prefer a comment that records a decision or a trap over one that narrates.
