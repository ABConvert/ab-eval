# Changelog

## Unreleased

First-run fixes, from setting the harness up on a private Python + Node monorepo from nothing.
See [docs/first-run.md](docs/first-run.md).

- **Python dependencies now reach the sandbox.** A `dep_dirs` entry names its `target`
  (`node_modules` by default, `.venv` for uv and poetry); a uv root used to install into a
  directory that was thrown away. The Python images put `/app/.venv` on `PATH`.
- **Runners list the `deps` they need**, so a Python case no longer installs (or fails on) npm
  roots. New `python312-node20` image for repositories with both suites.
- **Dependency installs:** `install_timeout` per root (default 3600s, was a fixed 1800s), a
  persistent download cache, progress output, and no leftover volumes after a failure. Images
  are rebuilt when their Dockerfile changes.
- **Curation has a command.** `collect` lists and saves the PRs awaiting a verdict; `curate`
  records them. A fresh data root no longer crashes on a missing `verdicts.json`.
- **Selection rules are settings** in a `selection:` block, and the collect report explains
  every rejection code.
- **doctor checks what it used to assume:** a down Docker daemon, GitHub access to each
  repository, and whether the Linear key reaches the team (`--offline` skips the network).
  Git worktrees are accepted. `linear_api_key_env` for more than one Linear workspace.
- **init reads git and CI:** files from `git ls-files` (no more nested worktrees), image named
  after the key, `npm ci` flags, pnpm vs npm, and pytest env from `.github/workflows`.
- **An empty GitHub answer is not cached**, and `validate` re-runs errored cases by default and
  labels any result it reads back from disk.

## 0.1.0 — first public release

Benchmark AI coding models on your own merged pull requests.

- **Cases from your own history.** `collect` mines merged pull requests and their tickets
  (GitHub Issues by default, Linear opt-in); `import swebench` brings in public SWE-bench
  variants for a first run without your own data.
- **Validation that checks the benchmark, not just the model.** `validate` proves each case's
  tests fail before the fix and pass with the team's own merged patch. A case its reference
  patch cannot pass is marked invalid, and `run` skips it without calling the model.
- **A sandbox with no network and no credentials.** Source arrives by `git archive`, tools run
  by `docker exec`, and your checkout is only ever read.
- **Providers:** Claude Code (subscription), Codex CLI, the Anthropic API, and any
  OpenAI-compatible endpoint, including local servers. A declared setting that a provider cannot
  send is refused at load rather than silently dropped.
- **Per-model run budgets** in `models.yaml`, overridable per run, recorded in every summary.
- **Scoring:** deterministic metrics plus an optional LLM code-quality judge. `--judge-key`
  picks the judge, `--write-as` keeps a second judge's scores beside the first, and a spend
  ledger with a hard limit caps what a priced judge may spend.
- **Dashboard:** cases, live runs, results, comparisons, a leaderboard, and a Setup page that
  checks the machine and edits the configuration (loopback only, never stores a key).
