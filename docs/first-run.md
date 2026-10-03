# Your first run on your own repository

The README's three commands get you a configuration. This is the rest of the path to a first
validated case, in order, with what to check at each step and what goes wrong when you do not.
It was written by setting the harness up on a private Python + Node monorepo from nothing and
writing down every place it stopped.

## Before you start

- **Docker with room to work.** Give the Docker VM 16 GB or more. A case container is capped
  near 6 GB, the prep container that installs dependencies needs about as much, and anything
  else you run in Docker shares the same VM. At 8 GB with a few other containers up, the daemon
  can stop answering altogether: every `docker` command hangs, including the harness's.
- **A GitHub login that can see the repository.** `collect` asks `gh` for merged pull
  requests, and `gh` answers a search on a private repository it cannot see with an empty list,
  not an error. `eval-harness doctor` checks this; if it fails, `gh auth switch` to the right
  account or export `GH_TOKEN` for the shell that runs the harness.
- **A private data root.** Case files contain your source and your tickets. Point
  `ABEVAL_DATA_ROOT` at a directory outside this checkout, ideally its own private repository,
  and export it in every shell that runs the harness.

```bash
export ABEVAL_DATA_ROOT=~/eval-data/my-project
```

## 1. `init` — draft the configuration

```bash
uv run eval-harness init --repo ~/code/my-project --key my-project
```

`init` reads the files git knows about (tracked, plus untracked files that are not ignored),
so nested worktrees, build output and anything gitignored are left alone. A git worktree works
as well as a clone. It also reads `.github/workflows/` for how CI installs and runs the suites.

Open `$ABEVAL_DATA_ROOT/config/repos.yaml` and check every `CHECK`:

- **`dep_dirs`** — keep only the roots a benchmarked suite needs. Each one is installed into
  its own Docker volume, mounted at `<dir>/<target>`. `target` is where the install writes:
  `node_modules` for npm, pnpm and yarn (the default), `.venv` for uv and poetry. A Python root
  without `target: .venv` installs into a directory that is thrown away, and every test then
  runs against a bare interpreter.
- **`install`** — must succeed on a clean checkout of an old commit. Compare it with your CI
  (`init` copies `npm ci` flags it finds there). A large lockfile can take a while the first
  time: `install_timeout` (seconds, default 3600) is per root, and the download cache persists
  between attempts, so a retry after a timeout resumes.
- **`runners`** — one per suite. `deps` lists the dep_dirs the suite needs; without it a case
  prepares every root, so a Python case waits on an npm install. `env` is set for every test
  command (`init` copies `KEY=value` prefixes it finds in front of `pytest` in CI). `full` is the
  whole-suite command `validate` uses for the regression baseline: make sure it runs offline,
  since the sandbox has no network — exclude suites that call external services. It runs once
  per case during `validate` and again after every attempt in `run`, so its length multiplies:
  a 26,000-test Python suite took 15 minutes a pass on 4 CPUs. Point it at the tests a change
  could plausibly break (`python -m pytest tests/unit -q`, say) rather than everything CI runs.
- **`image`** — `python312`, `node20`, or `python312-node20` for a repository with both. The
  image is rebuilt whenever its Dockerfile changes.

Then set a model with a key in `config/models.yaml` (the file names the environment variable,
never the key).

## 2. `doctor` — check everything, including the network

```bash
uv run eval-harness doctor
```

From the command line, doctor also makes two round trips the dashboard's Setup page does not:
it asks `gh` whether it can see each repository, and asks Linear whether the key reaches the
configured team. The second one matters when you work in more than one Linear workspace: a
valid key from the other workspace used to pass as "set" and fail at collect time as "ticket
not found". Give such a repository its own variable:

```yaml
my-project:
  linear_team: ENG
  linear_api_key_env: ENG_LINEAR_API_KEY
```

`doctor --offline` skips the round trips.

## 3. `collect --dry-run` — see what your history yields

```bash
uv run eval-harness collect --repo my-project --since 2026-03-01 --until 2026-09-01 --dry-run
```

The report is a funnel: merged PRs, how many name a ticket, how many each rule rejected, and
how many passed. Expect most PRs to fall out. Every rejection code is explained in the report:

| Code | Rejects a PR when | Setting |
|---|---|---|
| S1 | it was reverted within 30 days | |
| S2 | it names no ticket, the ticket is missing, or its text is shorter than the limit | `min_description` (200) |
| S3 | it has no test file or no code file | |
| S4 | it touches too many files or lines | `max_files` (10), `max_lines` (400) |
| S5 | it is a dependency bump, generated files, or config only | |
| S6 | its ticket has more merged PRs than the limit | `max_prs_per_ticket` (1) |
| R1 | its ticket has sub-issues | |
| R2 / R3 | it names more than one ticket, or its branch and title disagree | |
| R4 | its ticket was created after the PR opened | |
| R7 / R8 | it has too few test lines or code lines | `min_test_lines` (10), `min_code_lines` (8) |

The settings live in a `selection:` block in the repository's entry. Loosen them knowingly: a
bigger PR is a harder case, and a ticket split across several PRs gives each case a prompt that
describes more than its patch does.

Below 70% of PRs naming a ticket, collect stops unless you pass `--force`: the cases would be
a biased sample of your work.

## 4. `curate` — decide which PRs become cases

A PR that passes the rules still needs a person to say it is a fair task. Collect lists them
and saves the list; `curate` records the decision:

```bash
uv run eval-harness curate --repo my-project                          # what is waiting
uv run eval-harness curate --repo my-project --accept ENG-12,ENG-40 --kind bug_fix
uv run eval-harness curate --repo my-project --reject ENG-51 --reason "ticket describes a different change"
```

Read each PR and its ticket before accepting it. Accepted keys go into tier `A` (or `--tier`),
rejected ones into tier `X` so they are not offered again. Verdicts live in
`$ABEVAL_DATA_ROOT/data/curation/verdicts.json`, which you can review and commit.

## 5. `collect` — write the cases

The same command without `--dry-run` writes a case for every key with a verdict in the tiers
you ask for (`--tiers A` by default). To try a single PR without curating, build it directly:

```bash
uv run eval-harness collect --repo my-project --pr 1234 --kind bug_fix
```

## 6. `validate` — prove each case measures the model, not the setup

```bash
uv run eval-harness validate
```

The first case builds the image and fills the dependency volumes; progress is printed as it
goes. Then, per case: the tests must fail at the base commit, the full suite runs for a
baseline, and your team's own merged patch must make the tests pass. A case that cannot pass
with its own patch is marked invalid and `run` will skip it without calling a model. A case that
ends in `error` is the harness failing to set it up; fix the configuration and validate again —
errored cases are re-run, and any result read back from disk says so. A case marked invalid
before you changed the configuration keeps that verdict until you pass `--recheck`.

## 7. `run`, `score`, `report`

```bash
uv run eval-harness run --model claude-sonnet-5
uv run eval-harness dashboard
```

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `merged PRs 0` for a window with merges | the `gh` login cannot see the repository | `doctor`; `gh auth switch` or `GH_TOKEN` |
| `S2 … not found in Linear` for most keys | the Linear key belongs to another workspace | `doctor`; `linear_api_key_env` |
| `install failed … npm: not found` | a Node root in a Python image | `python312-node20`, or `deps` on the Python runner |
| `install failed … timed out` | a large cold install | raise `install_timeout`; the retry resumes from the cache |
| `install failed … invalid wheel` or a hash mismatch | a download cut off mid-write (Docker died) | validate again: a failed install clears the download cache |
| tests fail with `ModuleNotFoundError` for installed packages | a Python root without `target: .venv` | add it |
| every `docker` command hangs | the Docker VM is out of memory | restart Docker; give it 16 GB; stop other containers |
| `invalid: tests pass before any fix` | the case's tests do not exercise the change | reject it in `curate` |
