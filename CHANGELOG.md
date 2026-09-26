# Changelog

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
