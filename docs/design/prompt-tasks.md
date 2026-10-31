# Prompt tasks: benchmarking models on a product's own LLM feature

Status: proposed (2026-10-31). First adopter: NexRex activity summaries (NEX-2416 / NEX-2422).

## Why

ab-eval benchmarks coding agents: a case is a merged pull request and the grade is the team's own tests. Teams also run LLM features in
production (summaries, classifiers, extractors), and the question there is the same one ab-eval answers for code: *which model, at what
cost and speed, does this job best on our own data?* The pieces that answer it already exist here: the model registry (`models.yaml`,
`ModelConfig`), the provider adapters and the one-shot call, pricing and the spend ledger, paired statistics, compare, leaderboard and
dashboard. What is coding-specific is the case schema, the Docker runner and the test-based score.

A **prompt task** is a case whose input is a frozen prompt (system + user) produced by the product's own code, whose output is one model
completion, and whose grade comes from a rubric judge plus an optional product-owned deterministic scorer.

Non-goals: multi-turn agents for prompt tasks (the agent loop already covers that); building prompts inside ab-eval (the product
freezes them, so ab-eval never imports product code or its dependencies); hosting athlete or customer data in this repository.

## Case

`data/prompt-cases/<task>/<case_id>.json` under `ABEVAL_DATA_ROOT`:

```json
{
  "case_id": "nexrex-act-3f9c…",
  "task": "nexrex-activity-summary",
  "kind": "prompt",
  "system": "…exact system prompt…",
  "prompt": "…exact user prompt…",
  "prompt_version": "v2@sha256:…",
  "max_output_tokens": 2000,
  "grading_context": { "evidence": { "laps": [], "zones": {} }, "labels": { "type": "interval_run", "tier": "A_human" } },
  "strata": ["detected_ne_final", "paired_plan"],
  "split": "dev",
  "source": "export:nexrex-platform@<sha>",
  "created_at": "2026-10-31T00:00:00Z"
}
```

- `grading_context` is shown only to judges and scorers, never to the model under test.
- One task directory per product feature; `task.yaml` beside the cases names the rubric, the scorer command and default caps.
- Datasets and splits reuse `collect/datasets.py` and `collect/split.py` unchanged (they key on `case_id`).
- Import: `eval-harness import prompt-cases --task T --from DIR` validates against the pydantic model and copies; the producer
  (e.g. nr-platform's `python -m evals.activity export-prompt-cases`) writes the directory.

## Task config (`task.yaml`)

```yaml
task: nexrex-activity-summary
description: Daily AI activity summary, zh-TW/en/ja.
rubric: rubric.yaml               # judge dimensions, scales, instructions
scorer:                           # optional, product-owned deterministic checks
  command: ["uv", "run", "--project", "/srv/nr-platform", "python", "-m", "evals.activity", "score-output"]
  timeout_seconds: 60
judges: [claude-opus-5-5, codex-gpt-6-1-sol]   # keys in models.yaml
defaults: { max_output_tokens: 2000, repeats: 1 }
```

`rubric.yaml` declares dimensions (`name`, `scale` 1–5 or enum, `instruction`, `uses: [output, grading_context]`) so the judge prompt is
data, not code. The current code-quality judge becomes one rubric among others.

## Run

`eval-harness run --task T --model M [--dataset D] [--repeats K] [--concurrency N]`:

1. For each case × repeat: `oneshot(case.prompt, system=case.system, model=M, max_tokens=…)` — no Docker, no tools.
2. Record a `PromptAttempt` (`case_id`, `run_id`, `model`, `repeat`, `status`, `output_text`, `usage`, `cost_usd`, `latency_seconds`,
   `request_settings`, `error`, timestamps) to `results/<run_id>/attempts/<case_id>.<repeat>.json`; reuse/resume semantics as today
   (`reused` flag).
3. `RunMeta` gains `task` and `kind: prompt`.

Adapter change: `oneshot` returns `OneshotResult(text, usage, cost_usd, latency_seconds, request_settings)` instead of `str`
(existing callers take `.text`), honours `max_tokens` on every branch it can, and gains a `codex-cli` branch (read-only sandbox,
`--skip-git-repo-check`, empty cwd, model pinned) so subscription CLIs can be both candidates and judges.

## Score

`eval-harness score --run R` for a prompt run:

1. **Scorer** (if configured): one subprocess per attempt; stdin = `{case, output_text}` JSON, stdout = `{metrics: {name: number|bool},
   flags: [str]}`. Non-zero exit or invalid JSON → attempt marked `score_error`, never silently zero. Runs in the product's own
   environment, which is how product checks reuse product code without ab-eval importing it.
2. **Rubric judges**: each judge in `task.yaml` scores each attempt independently; results cached by
   (case, repeat, run, judge, rubric hash). Judges see `output_text` + `grading_context`, never other judges' scores. The spend ledger
   and `ABEVAL_SPEND_LIMIT_USD` apply.
3. **Summary**: `PromptCaseScore` (scorer metrics, per-judge rubric scores, judge mean, judge disagreement, cost, tokens, latency) and
   `RunSummary.metrics` with per-dimension means, bootstrap CIs, per-stratum breakdowns, judge–judge agreement (Cohen's κ for enums,
   weighted κ for scales) and repeat self-consistency when `repeats > 1`.
4. A judge that is also the model under test is flagged in the summary (self-preference), as the existing judge guidance asks.

## Compare, leaderboard, dashboard

- `compare --runs A --runs B` pairs attempts by `case_id` (and repeat) and reports paired-bootstrap differences per rubric dimension
  and scorer metric (`report/stats.py` as-is).
- Leaderboard for prompt tasks: quality = configurable weighted rubric/scorer composite; plus cost per case, p50/p90 latency, and
  quality per dollar. Only runs of the same task and dataset are ranked together.
- Dashboard: task selector; run view shows per-case output next to grading context and judge reasons; compare view as above.
- `export --run R --format summary-json` writes the run summary without case-level text, for publishing to a product's own admin page.

## Long-term benchmark instance

One always-on instance runs the product's benchmark on a schedule:

- Host: a private always-on machine with the product's checkout (for the scorer) and an ab-eval checkout, separate venvs.
  For NexRex: the `nexrex` homelab box (prompt tasks need no Docker; its local vLLM `qwen3-14b` can join as a cheap candidate).
- `ABEVAL_DATA_ROOT=/data/eval/<product>`; keys from the host's secret store via `api_key_env`; spend ledger per month with a hard limit.
- Cadence: weekly `import` of newly exported cases (the product's intake loop) → `run` for the pinned production model + the candidate
  list → `score` → `export` summary to the product. On demand: a prompt/model change runs the same dataset for a head-to-head.
- Dashboard bound to localhost / the tailnet only.

## Build order (each a PR with tests; unit suite stays Docker/network-free)

1. `OneshotResult` + `max_tokens` on all branches + `codex-cli` one-shot branch (fakes for each provider).
2. Prompt case model, `task.yaml` / `rubric.yaml` config, `import prompt-cases`, datasets/splits over prompt cases.
3. Prompt runner + `PromptAttempt` records + resume.
4. Scorer subprocess protocol + rubric judge + caching + ledger.
5. Prompt summary metrics (CIs, strata, κ, self-consistency) + compare.
6. Leaderboard + dashboard task views + `export summary-json`.
7. `docs/prompt-tasks.md` user guide + an example task in `tests/fixtures/` with a fake scorer.

## Open questions

- Whether the code-quality judge should migrate onto the rubric engine now or later (later; keep it working unchanged).
- Rebase with PR #7 (`fix/pre-launch`), which touches `report/render.py`.
