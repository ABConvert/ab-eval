# AGENTS.md

Instructions for a coding agent working with ab-eval. Two jobs come up; find yours, then read
only that section and what it links to.

- **A. Set ab-eval up against someone's repository** and produce validated cases.
- **B. Change ab-eval itself** (a fix, a runner, a provider).

Humans start at [README.md](README.md). The step-by-step first run is
[docs/first-run.md](docs/first-run.md); this file is the short, rule-shaped version of it.

## Ground rules for either job

- **Never write a secret anywhere.** Config names environment variables (`api_key_env`,
  `linear_api_key_env`); values stay in the shell. Do not put keys in `repos.yaml`,
  `models.yaml`, `.env` files you commit, case files, commit messages or logs.
- **Case data is private.** Cases contain the target repository's source and ticket text.
  They live under `ABEVAL_DATA_ROOT`, outside this checkout. Never commit anything under
  `data/` or `results/` here (both are gitignored for that reason).
- **The target repository is read-only.** The harness reads it with `git archive`; you should
  not check out, commit, or run commands in it on the harness's behalf either.
- **Docker is shared.** Long runs share one VM with the user's other containers. Check
  `docker info` answers before a long run; if every `docker` command hangs, stop and tell the
  user rather than restarting their Docker Desktop.

## A. Setting up against a repository

Work in this order. Each step has a check; do not move on until it passes.

```bash
export ABEVAL_DATA_ROOT=~/eval-data/<project>        # private; ask the user where
uv sync
uv run eval-harness init --repo <path> --key <project>
uv run eval-harness doctor                           # must end "ready" or only warnings
uv run eval-harness collect --repo <project> --since YYYY-MM-DD --until YYYY-MM-DD --dry-run
uv run eval-harness curate --repo <project>          # lists what awaits a verdict
uv run eval-harness validate --case <ID>             # start with ONE case
```

**After `init`, review `$ABEVAL_DATA_ROOT/config/repos.yaml` against the repository's CI**
(`.github/workflows/`). Every `CHECK` is a guess. The things that break a first run:

| Field | Must be true |
|---|---|
| `dep_dirs[].target` | `.venv` for uv/poetry roots, `node_modules` (default) for npm/pnpm/yarn |
| `dep_dirs[].install` | the command CI uses, succeeding on a clean checkout of an *old* commit |
| `runners.*.deps` | lists only the dep_dirs that suite needs |
| `runners.*.env` | what CI sets for tests (e.g. `APPLICATION_ENV=test`) |
| `runners.*.full` | runs **offline** (the sandbox has no network) and in reasonable time: it runs once per case in validate and after every attempt in run |
| `image` | `python312-node20` when the repo has both a Python and a Node suite |

**`doctor` failures and what they mean:**

- `github <key>` fails → the `gh` login cannot see the repository. `collect` would silently
  find 0 PRs. Ask the user to `gh auth switch` or export `GH_TOKEN`; do not guess accounts.
- `tickets <key>` fails with "no team" → the Linear key is from another workspace. Set
  `linear_api_key_env` to the variable that holds the right key, and ask the user for it.

**Curation is a human decision.** `curate --accept` records that a PR is a fair task. Do not
accept PRs in bulk (`--all`) unless the user told you to; list them and ask, or read each PR
and its ticket and say why you accept or reject it. To try one PR without curating, use
`collect --repo <project> --pr <N> --kind bug_fix|feature`.

**Reading `validate`:**

- `completed` — the case's tests fail before the fix and pass with the team's patch. Good.
- `invalid` — the case itself is unusable (tests pass before the fix, or the reference patch
  cannot pass). Reject it in `curate`; do not change harness code to make it pass.
- `error` — the harness could not set the case up. Read the message, fix `repos.yaml`, and
  run `validate` again; errored cases are retried automatically. If a case was marked invalid
  before a config fix, `validate --recheck` re-checks it.

The troubleshooting table at the end of [docs/first-run.md](docs/first-run.md) maps symptoms
to causes. Report to the user: the collect funnel numbers, which cases validated, and anything
you changed in `repos.yaml` and why.

## B. Changing ab-eval

```bash
uv sync
uv run pytest -q                 # unit tests; no Docker, no network
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m docker          # integration tests; need Docker (and network for installs)
```

All four of the first commands must pass before a commit; CI runs exactly those.

**Layout**

| Path | What lives there |
|---|---|
| `src/eval_harness/cli.py` | every command (`typer`) |
| `src/eval_harness/config.py` | `repos.yaml` / `models.yaml` models; new fields go here |
| `src/eval_harness/scaffold.py` | `init`: reads a repository and its CI, drafts `repos.yaml` |
| `src/eval_harness/doctor.py` | checks; `online=True` adds GitHub/Linear round trips (CLI only) |
| `src/eval_harness/collect/` | PR mining, ticket sources, selection rules (`filters.py`), curation |
| `src/eval_harness/harness/` | sandbox (`sandbox.py`), dependency prep (`deps.py`), case loop (`runner.py`), test commands and parsers (`testrun.py`) |
| `src/eval_harness/adapters/` | model providers |
| `src/eval_harness/dashboard/` | local UI; runs `doctor` offline on every Setup render |
| `docker/` | base images; changing one rebuilds existing images automatically |
| `tests/fixtures/` | the fixture repository (`demo-app`, vitest) the unit tests configure against |

**Rules the codebase holds itself to** (from [CONTRIBUTING.md](CONTRIBUTING.md)):

- **A test that fails without the change.** For a bug, write the failing test first. Then
  check it: revert the fix, watch the test go red, restore it. Several tests here once passed
  vacuously (a thread that raised; a fake missing a method) — unhandled thread exceptions now
  fail the suite, so do not silence them.
- **Fakes must match the real shape.** Use the real pydantic models (`DepDir`, `RepoConfig`,
  `Case`) in tests, not `SimpleNamespace`; a new field then cannot slip past a test.
- **New config fields default to today's behaviour**, so every existing `repos.yaml` keeps
  working. Validate cross-field references in a `model_validator` (see `_runner_deps_exist`).
- **Tests never read the operator's config or data.** `tests/conftest.py` points
  `ABEVAL_DATA_ROOT` at the fixtures; a test that writes (curation, results) must point it at
  `tmp_path` first and call `paths.reset_cache()`.
- **Nothing in the unit suite calls Docker, GitHub or Linear.** Inject fakes (`doctor._run`,
  `linear.probe_team`, a recording Docker). Real calls belong under `@pytest.mark.docker`.
- **Comments explain why** — a decision or a trap — not what the code says.
- **Measured claims.** "Faster" or "fixes X" wants a number or a log line.

**Where the traps are**

- Commands in containers run through `sh -lc`, a login shell: Debian's `/etc/profile` resets
  `PATH`, so anything the images put on `PATH` must also go in `/etc/profile.d`.
- Dependency volumes mount at `<dir>/<target>`. An install that writes anywhere else is lost
  when the prep container exits.
- `gh` answers a search on a repository it cannot see with `[]`, not an error. Never cache an
  empty result as if it were data.
- A result read back from disk must say so (`AttemptRecord.reused`); a stale error printed as
  fresh sends the user to fix something they already fixed.
- The unit fixture repository has no Python suite. A change to the pytest path needs the
  Docker integration test (`tests/integration/test_python_case.py`) run locally.

**Commits** are conventional (`fix(deps): …`, `docs: …`), one area per commit, and the body
says what broke and how it was found. Update the `Unreleased` section of
[CHANGELOG.md](CHANGELOG.md) for anything a user would notice. A pull request lists behaviour
changes a reviewer must decide on, and what it deliberately did not do.
