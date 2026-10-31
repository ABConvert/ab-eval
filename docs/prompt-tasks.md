# Prompt tasks: benchmark models on your own LLM feature

ab-eval's coding benchmark answers "which model fixes our bugs best, at what cost?". A **prompt
task** answers the same question for an LLM feature you already run in production, such as a
summary, a classifier or an extractor:

- A **case** is one frozen prompt (system + user) that your product's own code produced.
- An **attempt** is one completion from the candidate model.
- The **grade** comes from rubric judges, plus an optional scorer that you own.

There is no Docker and no tool use. The design and its reasoning are in
[design/prompt-tasks.md](design/prompt-tasks.md).

```bash
export ABEVAL_DATA_ROOT=~/eval-data/my-product      # private: cases hold your users' data
uv run eval-harness import prompt-cases --task my-summary --from /path/to/export
uv run eval-harness run --task my-summary --model claude-opus-5-5 --dataset holdout
uv run eval-harness run --task my-summary --model gpt-6.1 --dataset holdout
uv run eval-harness compare --runs <run-a> --runs <run-b>
uv run eval-harness leaderboard --task my-summary
uv run eval-harness export --run <run-a> --format summary-json
```

`run` scores the run when it finishes, unless you pass `--no-score`. To re-score, use
`eval-harness score --run R`.

## 1. Export cases from your product

ab-eval never imports your product's code. Your product writes one JSON file per case into a
directory, and puts `task.yaml` and `rubric.yaml` beside them:

```json
{
  "case_id": "act-3f9c21",
  "task": "my-summary",
  "kind": "prompt",
  "system": "…the exact system prompt your product sends…",
  "prompt": "…the exact user prompt…",
  "prompt_version": "v2@sha256:…",
  "max_output_tokens": 2000,
  "grading_context": {"evidence": {"laps": []}, "labels": {"type": "interval_run"}},
  "strata": ["paired_plan"],
  "split": "dev",
  "source": "export:my-product@<sha>",
  "created_at": "2026-10-31T00:00:00Z"
}
```

- **`grading_context` is shown to judges and scorers only, never to the model under test.** Put
  ground truth here, such as the evidence the output must be faithful to, or gold labels.
- **`case_id` becomes a file name.** Use letters, digits, `.`, `_` and `-`, and make it opaque:
  hash your internal ids rather than using names.
- **`strata` are free-form labels.** The summary breaks every number down by stratum and by split.
- **Unknown fields are refused.** A misspelt `grading_context` would otherwise give every judge
  an empty dict.

Import validates every file, and imports all of them or none:

```bash
uv run eval-harness import prompt-cases --task my-summary --from ./export [--holdout-fraction 0.3]
```

`--holdout-fraction` assigns `dev`/`holdout` by `created_at` to cases that have no `split`.
Re-importing reports cases whose content changed under an existing id. Runs made before the
change answered the old prompt, and each attempt records the `prompt_version` it answered.

Datasets work as they do for coding cases. Pass `--task` so that the built-in `all`, `dev` and
`holdout` mean this task's cases:

```bash
uv run eval-harness dataset list --task my-summary
uv run eval-harness dataset create --name pilot --cases act-3f9c21,act-77a0b1
```

## 2. `task.yaml`

```yaml
task: my-summary
description: Daily AI activity summary.
rubric: rubric.yaml                 # beside this file
scorer:                             # optional
  command: ["uv", "run", "--project", "/srv/my-product", "python", "-m", "evals.score_output"]
  timeout_seconds: 60
judges: [claude-opus-5-5, codex-gpt-6-1-sol]   # keys in models.yaml
defaults: {max_output_tokens: 2000, repeats: 1}
quality_metrics: {numbers_match: 1.0}          # scorer metrics in the quality composite
```

The output ceiling for an attempt comes from the first of these that is set: the case's
`max_output_tokens`, then `defaults.max_output_tokens`, then the model's own `max_output_tokens`.

## 3. `rubric.yaml`

The judge prompt is assembled from this file, so a reviewer can read exactly what judges are
asked. Its hash keys the judge cache.

```yaml
name: my-summary-v1
instructions: Grade the summary only against the evidence; do not reward length.
dimensions:
  - name: faithfulness
    scale: "1-5"
    instruction: 5 = every number is supported by the evidence; 1 = it contradicts it.
    uses: [output, grading_context]
  - name: type_correct
    scale: ["yes", "no"]
    good: ["yes"]
    instruction: "yes" if the summary describes the workout as the label says.
    uses: [output, grading_context]
```

- **`scale`** is either `"LO-HI"` (integers) or a list of labels.
- **`uses`** controls what the judge sees for a dimension. The output is always shown. Add
  `grading_context` for fidelity, and `prompt` to judge instruction-following.
- **`weight`** (default 1) sets the dimension's share of the quality composite. An enum dimension
  counts only if it lists `good` labels.

## 4. The scorer protocol

The scorer is optional. It holds your product's own deterministic checks, run once per attempt
in your product's environment, so it can reuse your product's code.

```
stdin   {"case": {...the whole case, grading_context included...}, "output_text": "..."}
stdout  {"metrics": {"numbers_match": true, "word_count": 312}, "flags": ["mentions_hr_zone_5"]}
```

- **Metrics are numbers or booleans.** Strings are rejected.
- **A failure is never scored as zero.** A non-zero exit, output that is not JSON, a wrong type,
  or a timeout marks the attempt `score_error`.
- **Example:** `tests/fixtures/prompt-tasks/fake_scorer.py`.

## 5. Running

```bash
uv run eval-harness run --task T --model M [--dataset D | --case ID,ID] [--repeats K] [--concurrency N]
```

- **One call per attempt.** Each (case, repeat) is one `oneshot()` call, and each is saved to
  `results/<run>/attempts/<case>.<repeat>.json`.
- **Re-running the command resumes the run.** Completed attempts are reused, never regenerated.
  Errors re-run only with `--retry-errors`.
- **Any provider in `models.yaml` can be a candidate or a judge.** That includes the
  subscription CLIs: `claude-code`, and `codex-cli`, which runs hermetically in an empty temp
  directory with `--sandbox read-only` and the model pinned.
- **Not every provider can cap output.** `codex exec` has no output ceiling, so the attempt
  records the cap as "not a parameter".

**Spend.** Every priced call (candidate or judge) is appended to the spend ledger. With
`ABEVAL_SPEND_LIMIT_USD` set, a call that would start past the limit is not made:

- A **run** stops, and the same command resumes it.
- **Scoring** stops and writes no summary.
- **Where the ledger lives:** without `ABEVAL_SPEND_LEDGER`, prompt-task commands use
  `results/_spend/YYYY-MM.jsonl`, which gives a monthly budget.
- **Subscription CLIs** report a notional cost that counts toward the limit unless the model has
  prices in `models.yaml`.

## 6. Scoring and the summary

`score` writes `results/<run>/prompt-summary.json`.

**Per attempt** (`PromptCaseScore`):

- scorer metrics and flags;
- each judge's score and reason per dimension;
- the judge mean, and the spread between judges;
- cost, tokens and latency.

Statuses: `completed`, `error` (the model call failed) and `score_error` (the scorer failed, or
every judge did). Neither failure status enters a mean.

**Judgements are cached** by run, judge, case and repeat, together with the rubric hash and the
hash of the output graded. Editing the rubric, or regenerating an attempt, re-asks exactly what
changed. `--no-cache` re-asks everything; `--judge K` overrides `task.yaml`'s judges.

**Run metrics** (`metrics` in the summary):

- **Quality composite (0..1):** the weighted mean of the normalised rubric dimensions and
  `quality_metrics`, with a bootstrap CI.
- **Per rubric dimension:** the mean with a bootstrap CI and n. Enum dimensions also get their
  label distribution.
- **Per scorer metric:** the mean (or rate, for booleans) with a CI.
- **Per stratum and per split:** n, quality, and the dimension means.
- **Judge agreement:** Cohen's κ for enum dimensions and quadratic-weighted κ for scales, with
  raw exact agreement. Low κ means judging noise may be larger than the differences between
  models.
- **Repeat self-consistency (`--repeats > 1`):** the mean absolute difference between repeats of
  the same case.
- **Cost, tokens and latency:** cost per case and per attempt, judge spend, p50/p90 latency,
  truncation count (`stop_reason` of `max_tokens`/`length`), and quality per dollar.
- **Self-preference:** any judge whose model is the candidate's own is listed. Its scores are
  kept but flagged; prefer judges from another provider.

## 7. Compare, leaderboard, dashboard, export

- **`compare --runs A --runs B [...]`** pairs attempts by case and repeat, then reports
  paired-bootstrap differences per dimension, per scorer metric and for quality, against the
  first run. It writes `results/compare-prompt-*.{json,md}`.
- **`leaderboard --task T [--dataset D]`** ranks by quality, next to cost per case, p50 latency
  and error rate. Runs are ranked together only if they share a task and dataset.
- **Dashboard → Prompt tasks** has a task selector, the leaderboard and run list, and a compare
  view. Its run view shows each output next to its grading context, the judges' scores and
  reasons, and scorer results.
- **`export --run R --format summary-json`** writes a version for a product's admin page. It
  includes metrics and per-case numbers, but no outputs, judge reasons, error text or rubric
  instructions.

## Privacy

Cases and results contain your users' data. Keep `ABEVAL_DATA_ROOT` outside this checkout.

Judges and candidates are third-party models, so sending cases to them is a data-sharing
decision for your product. Make it before the first run.
