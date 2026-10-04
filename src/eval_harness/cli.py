from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from eval_harness import PROJECT_ROOT

app = typer.Typer(
    help="Replay merged engineering work against candidate models.", no_args_is_help=True
)


@app.callback()
def _root() -> None:
    """eval-harness command group."""


@app.command()
def collect(
    repo: str = typer.Option(..., help="Key in config/repos.yaml"),
    pr: int | None = typer.Option(None, help="Build a single case from one PR (Step 0 path)"),
    kind: str | None = typer.Option(None, help="bug_fix or feature (single-PR path only)"),
    since: str | None = typer.Option(None, help="YYYY-MM-DD; default six months ago"),
    until: str | None = typer.Option(None, help="YYYY-MM-DD; default today"),
    tiers: str = typer.Option("A", help="Comma-separated curation tiers to include"),
    tier: str = typer.Option("A", help="Tier recorded on a single-PR case"),
    refresh: bool = typer.Option(
        False, help="Re-fetch GitHub and Linear instead of using the cache"
    ),
    dry_run: bool = typer.Option(False, help="Report only; write no case files"),
    force: bool = typer.Option(False, help="Proceed even when the match rate is below 70%"),
) -> None:
    """Mine merged PRs into case files, or build one case from a PR number."""
    from eval_harness.config import load_models, load_repos

    repo_cfg = load_repos()[repo]
    if pr is not None:
        from eval_harness.collect.cases import save_case
        from eval_harness.collect.single import build_case

        if kind is None:
            raise typer.BadParameter("--kind is required with --pr")
        case = build_case(repo_cfg, pr, tier=tier, kind=kind, kind_source="review")
        path = save_case(case)
        typer.echo(
            f"wrote {path} ({case.case_id}: {case.files_changed} files, "
            f"{case.lines_changed} lines, runner={case.test_runner})"
        )
        return

    from datetime import UTC, datetime, timedelta

    from eval_harness.collect.pipeline import collect as run_collect
    from eval_harness.collect.report import render_text, write_report

    today = datetime.now(UTC).date()
    since_s = since or (today - timedelta(days=182)).isoformat()
    until_s = until or today.isoformat()
    report = run_collect(
        repo_cfg,
        since=since_s,
        until=until_s,
        tiers=tuple(t.strip() for t in tiers.split(",") if t.strip()),
        refresh=refresh,
        dry_run=dry_run,
        classifier_model=load_models().get("classifier"),
    )
    typer.echo(render_text(report))
    if not dry_run:
        typer.echo(f"report: {write_report(report)}")
    if report.match.rate < 0.70 and not force:
        typer.echo(
            "Fewer than 70% of merged PRs name a ticket, so the cases would be a biased "
            "sample of your work. Pass --force to proceed anyway."
        )
        raise typer.Exit(code=2)
    raise typer.Exit(code=1 if report.secrets_hits else 0)


def _reused_note(r: Any) -> str:
    """Say so when a line is a saved result: after a config fix it looks like a fresh failure."""
    if not getattr(r, "reused", False):
        return ""
    when = r.finished_at or "an earlier run"
    return f"  (saved result from {when}; not re-run" + (
        " — pass --retry-errors to re-run errors)" if r.status == "error" else ")"
    )


@app.command()
def curate(
    repo: str = typer.Option(..., help="Key in config/repos.yaml"),
    accept: str | None = typer.Option(None, help="Comma-separated ticket keys to accept"),
    reject: str | None = typer.Option(None, help="Comma-separated ticket keys to reject"),
    all_pending: bool = typer.Option(False, "--all", help="Accept everything pending"),
    tier: str = typer.Option("A", help="Tier recorded for accepted keys"),
    kind: str = typer.Option("-", help="bug_fix or feature for accepted keys; - to classify"),
    reason: str = typer.Option("", help="Why, recorded beside the verdict"),
) -> None:
    """Review the PRs the last collect left pending, and record verdicts for them.

    Without --accept, --reject or --all it lists what is pending. collect writes a case only
    for keys with a verdict in a tier it was asked for (--tiers, default A); a rejected key
    is recorded in tier X so the next collect does not ask again.
    """
    from eval_harness.collect.curation import load_pending, load_verdicts, record_verdicts
    from eval_harness.paths import curation_path

    if kind not in ("-", "bug_fix", "feature"):
        raise typer.BadParameter("--kind is bug_fix, feature or -")
    pending = load_pending(repo)
    done = load_verdicts()
    todo = [c for c in pending if c.key not in done]
    if not (accept or reject or all_pending):
        if not pending:
            typer.echo(f"nothing pending for {repo}: run eval-harness collect --repo {repo} first")
            return
        for c in pending:
            v = done.get(c.key)
            mark = f"[{v.tier}]" if v else "[ ]"
            typer.echo(
                f"{mark:5} {c.key:12} #{c.pr:<6} {c.files:2} files {c.lines:4} lines  "
                f"{c.title[:70]}"
            )
        typer.echo(
            f"\n{len(todo)} of {len(pending)} without a verdict. "
            "Read each PR and its ticket, then:\n"
            f"  eval-harness curate --repo {repo} --accept KEY,KEY [--kind bug_fix|feature]\n"
            f"  eval-harness curate --repo {repo} --reject KEY --reason 'why'"
        )
        return

    def keys(value: str | None) -> list[str]:
        return [k.strip().upper() for k in (value or "").split(",") if k.strip()]

    unknown: list[str] = []
    to_accept = [c.key for c in todo] if all_pending else keys(accept)
    if to_accept:
        unknown += record_verdicts(
            repo, to_accept, tier=tier, reason=reason or "accepted", kind=kind
        )
    if reject:
        unknown += record_verdicts(repo, keys(reject), tier="X", reason=reason or "rejected")
    if unknown:
        typer.echo(f"not in the pending list for {repo}, so not recorded: {', '.join(unknown)}")
    typer.echo(
        f"verdicts: {curation_path()}\n"
        f"next: eval-harness collect --repo {repo} with the same --since/--until"
    )
    raise typer.Exit(code=1 if unknown else 0)


@app.command()
def validate(
    case: str | None = typer.Option(None, help="Case id; default all cases"),
    split: str = typer.Option("all", help="dev, holdout, or all (when --case is not given)"),
    concurrency: int = typer.Option(
        1, help="Containers at once; >1 needs a Docker VM with more than 8 GB"
    ),
    # On by default here, unlike `run`: an error in validate is the harness failing to set a
    # case up, and the next thing anyone does after fixing repos.yaml is validate again.
    retry_errors: bool = typer.Option(
        True, help="Re-run cases whose saved validation ended in an error"
    ),
    recheck: bool = typer.Option(
        False, help="Re-validate every selected case, ignoring saved results"
    ),
) -> None:
    """Check cases build at their base commit, fail before any fix, and record the baseline."""
    import asyncio

    from eval_harness.adapters.base import Caps
    from eval_harness.adapters.fake import NoopAdapter
    from eval_harness.collect.cases import load_case, save_case
    from eval_harness.config import load_repos
    from eval_harness.harness.runner import run_many

    cases = [load_case(case)] if case else _cases_for_split(split)
    if not cases:
        typer.echo("no cases selected")
        raise typer.Exit(code=1)
    repo = next(r for r in load_repos().values() if r.github == cases[0].repo)
    recs = asyncio.run(
        run_many(
            cases,
            repo,
            adapter=NoopAdapter(),
            caps=Caps(),
            run_id="validate",
            concurrency=concurrency,
            retry_errors=retry_errors,
            validate_only=True,
            recheck=recheck,
        )
    )
    by_id = {c.case_id: c for c in cases}
    counts = {"completed": 0, "invalid": 0, "error": 0}
    for r in recs:
        counts[r.status] = counts.get(r.status, 0) + 1
        tb = r.tests_before
        fs = r.full_suite
        if r.status == "completed" and fs is not None:
            c = by_id[r.case_id]
            c.full_suite_baseline_failures = fs.failed_tests
            save_case(c)
        typer.echo(
            f"{r.case_id}: {r.status} | before: {tb.failed if tb else '?'} failed of "
            f"{tb.total if tb else '?'} | full-suite baseline: {fs.failed if fs else '?'} "
            f"failing of {fs.total if fs else '?'} | {r.wall_clock_seconds}s {r.error or ''}"
            + _reused_note(r)
        )
    typer.echo(f"validate: {counts}")
    raise typer.Exit(code=0 if counts["invalid"] == 0 and counts["error"] == 0 else 1)


def _cases_for_split(split: str) -> list:  # type: ignore[type-arg]
    """Cases in a dataset. `dev`, `holdout` and `all` are built in; anything else is saved."""
    from eval_harness.collect.cases import load_case
    from eval_harness.collect.datasets import load as load_dataset

    try:
        ds = load_dataset(split)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e)) from e
    return [load_case(i) for i in ds.case_ids]


@app.command()
def dataset(
    action: str = typer.Argument(..., help="list, show, create, add, remove or delete"),
    name: str = typer.Option("", help="Dataset name"),
    cases: str = typer.Option("", help="Comma-separated case ids"),
    ticket: str = typer.Option("", help="Linear ticket key; collects its PR if needed"),
    description: str = typer.Option("", help="What this dataset is for"),
    kind: str = typer.Option("bug_fix", help="Kind recorded when collecting a new case"),
) -> None:
    """Create and edit named sets of cases."""
    from eval_harness.collect import datasets as ds_mod

    ids = [c.strip() for c in cases.split(",") if c.strip()]
    if action == "list":
        for d in ds_mod.load_all():
            mark = " (built-in)" if d.built_in else ""
            typer.echo(f"{d.name}{mark}: {len(d)} case(s) {d.description}")
        return
    if action == "show":
        d = ds_mod.load(name)
        typer.echo(f"{d.name}: {len(d)} case(s)\n" + "\n".join(d.case_ids))
        return
    if action == "create":
        d = ds_mod.create(name, description, ids)
        typer.echo(f"created {d.name} with {len(d)} case(s)")
        return
    if action == "delete":
        ds_mod.delete(name)
        typer.echo(f"deleted {name}")
        return
    if action == "add":
        if ticket:
            ids.append(_collect_ticket(ticket, kind))
        d = ds_mod.add_cases(name, ids)
        typer.echo(f"{d.name} now has {len(d)} case(s)")
        return
    if action == "remove":
        d = ds_mod.remove_cases(name, ids)
        typer.echo(f"{d.name} now has {len(d)} case(s)")
        return
    raise typer.BadParameter(f"unknown action {action!r}")


def _collect_ticket(ticket: str, kind: str) -> str:
    """Case id for a Linear ticket, collecting it from its merged PR when it is new."""
    from eval_harness.collect.cases import save_case
    from eval_harness.collect.single import build_case
    from eval_harness.config import load_repos
    from eval_harness.paths import cases_dir

    key = ticket.strip().upper()
    if (cases_dir() / f"{key}.json").exists():
        return key
    repo = next(iter(load_repos().values()))
    pr = _pr_for_ticket(repo, key)
    if pr is None:
        raise typer.BadParameter(
            f"no merged PR in the cache carries {key}; run collect --refresh, "
            "or add the case by PR number"
        )
    case = build_case(repo, pr, tier="A", kind=kind, kind_source="manual")
    save_case(case)
    typer.echo(f"collected {key} from PR #{pr}")
    return key


def _pr_for_ticket(repo: object, key: str) -> int | None:
    from pathlib import Path

    from eval_harness.collect.github import PullRequest, load_cache
    from eval_harness.collect.join import ticket_for
    from eval_harness.paths import raw_cache

    cache = raw_cache(str(getattr(repo, "key", "")))
    if not cache.is_dir():
        return None
    best: int | None = None
    for path in sorted(Path(cache).glob("prs-*.json")):
        for pr in load_cache(cache, path.stem, PullRequest) or []:
            found, _ = ticket_for(pr, getattr(repo, "linear_team", ""))
            if found == key and (best is None or pr.number > best):
                best = pr.number
    return best


@app.command()
def review_tests(
    case: str = typer.Option(..., help="Case id, or comma-separated ids"),
) -> None:
    """Ask the judge whether a case's tests actually pin the ticket's behaviour."""
    from eval_harness.config import load_models
    from eval_harness.metrics.test_quality import review_case

    judge = load_models()["judge"]
    for cid in [c.strip() for c in case.split(",") if c.strip()]:
        r = review_case(cid, judge)
        if r.error:
            typer.echo(f"{cid}: error {r.error}")
        else:
            typer.echo(f"{cid}: {r.verdict} ({r.score}) {r.reason[:120]}")


@app.command()
def run(
    model: str = typer.Option(..., help="Key in config/models.yaml, 'noop', or 'human-patch'"),
    case: str | None = typer.Option(None, help="Case id, or comma-separated ids; default: --split"),
    split: str = typer.Option(
        "holdout", help="Dataset name: dev, holdout, all, or a saved one (when --case is absent)"
    ),
    run_id: str | None = typer.Option(None),
    concurrency: int = typer.Option(
        1, help="Containers at once; >1 needs a Docker VM with more than 8 GB"
    ),
    # Every cap defaults to None, not to its number: a flag that always carries a value is
    # a flag that always wins, and the model's own `caps:` block would never be read.
    max_turns: int | None = typer.Option(None, help="Override the model's caps.max_turns (40)"),
    wall_clock: int | None = typer.Option(
        None, help="Seconds per case; overrides caps.wall_clock_seconds (1800)"
    ),
    max_output_tokens_total: int | None = typer.Option(
        None, help="Output tokens per case; overrides caps.max_output_tokens_total (120000)"
    ),
    tool_timeout: int | None = typer.Option(
        None, help="Seconds for one tool call; overrides caps.tool_timeout_seconds (300)"
    ),
    retry_errors: bool = typer.Option(False),
    no_score: bool = typer.Option(False, help="Skip DeepEval scoring after the run"),
    round_id: str | None = typer.Option(None, help="Use a frozen benchmark round"),
) -> None:
    """Run a model against a case in the sandbox and record the attempt."""
    import asyncio
    from datetime import UTC, datetime

    from eval_harness.adapters.base import Caps
    from eval_harness.adapters.fake import HumanPatchAdapter
    from eval_harness.adapters.registry import make_adapter
    from eval_harness.collect.cases import load_case
    from eval_harness.config import load_models, load_repos
    from eval_harness.harness.record import RunMeta, save_run_meta
    from eval_harness.harness.runner import harness_git_sha, run_many

    frozen = None
    if round_id:
        from eval_harness import rounds

        if not run_id:
            raise typer.BadParameter("A round attempt requires --run-id")
        if any(
            v is not None for v in (max_turns, wall_clock, max_output_tokens_total, tool_timeout)
        ):
            raise typer.BadParameter("Round budgets are frozen and cannot be overridden")
        frozen, entry = rounds.execution(round_id, run_id)
        cases, repo, model = frozen.cases, frozen.repo, entry.model.key
        models = {model: entry.model}
        retry_errors = False
        concurrency = 1
    else:
        cases = (
            [load_case(x.strip()) for x in case.split(",") if x.strip()]
            if case
            else _cases_for_split(split)
        )
        if not cases:
            typer.echo("no cases selected")
            raise typer.Exit(code=1)
        repo = next(r for r in load_repos().values() if r.github == cases[0].repo)
        models = load_models()
    adapter = (
        HumanPatchAdapter({c.case_id: c.human_patch for c in cases})
        if model == "human-patch"
        else make_adapter(model, models)
    )
    declared = models[model].caps if model in models and models[model].caps else None
    caps = Caps.resolve(
        declared.model_dump() if declared else None,
        max_turns=max_turns,
        wall_clock_seconds=wall_clock,
        max_output_tokens_total=max_output_tokens_total,
        tool_timeout_seconds=tool_timeout,
    )
    if frozen:
        caps = frozen.caps
    rid = run_id or f"{model}-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}"
    snapshot = models[model].model_dump() if model in models else {"adapter": model}
    from eval_harness.collect.datasets import matching

    meta = RunMeta(
        run_id=rid,
        round_id=round_id,
        model=model,
        dataset=frozen.dataset
        if frozen
        else (None if case else split) or matching([c.case_id for c in cases]),
        model_config_snapshot=snapshot,
        caps=caps.model_dump(),
        concurrency=concurrency,
        harness_git_sha=harness_git_sha(),
        image=repo.image,
        cases=[c.case_id for c in cases],
        started_at=datetime.now(UTC).isoformat(),
    )
    save_run_meta(meta)
    recs = asyncio.run(
        run_many(
            cases,
            repo,
            adapter=adapter,
            caps=caps,
            run_id=rid,
            concurrency=concurrency,
            retry_errors=retry_errors,
        )
    )
    meta.finished_at = datetime.now(UTC).isoformat()
    save_run_meta(meta)
    for r in recs:
        ta = r.tests_after
        typer.echo(
            f"{r.case_id}: {r.status} resolved={r.resolved} | after: "
            f"{ta.passed if ta else '?'}/{ta.total if ta else '?'} "
            f"no_regression={r.no_regression} lint_ok={r.lint_ok}"
            f"{' TOUCHED_TESTS=' + ','.join(r.touched_test_files) if r.touched_test_files else ''}"
            f" | turns={r.turns} "
            f"tools={r.tool_calls} cost=${r.cost_usd:.3f} total_wall={r.wall_clock_seconds}s "
            f"agent_wall={r.agent_wall_clock_seconds}s "
            f"cap={r.cap_hit} {r.error or ''}" + _reused_note(r)
        )
    typer.echo(f"results: results/{rid}/")
    if not no_score:
        _score(rid, judge=True, cache=True)


def _score(
    run_id: str,
    *,
    judge: bool,
    cache: bool,
    judge_key: str = "judge",
    write_as: str | None = None,
) -> None:
    from eval_harness.config import load_models
    from eval_harness.metrics import dataset
    from eval_harness.report.summary import summary_path

    judge_cfg = None
    from eval_harness.harness.record import results_dir

    meta_path = results_dir(run_id) / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if judge and not meta.get("round_id"):
        models = load_models()
        if judge_key not in models:
            raise typer.BadParameter(
                f"no entry {judge_key!r} in models.yaml to judge with; "
                f"entries are: {', '.join(sorted(models))}"
            )
        judge_cfg = models[judge_key]
    from eval_harness import rounds
    from eval_harness.harness.runner import harness_git_sha

    frozen_cases = None
    if meta.get("round_id"):
        frozen = rounds.load(meta["round_id"])
        if (
            frozen.harness_sha != meta.get("harness_git_sha")
            or frozen.harness_sha != harness_git_sha()
        ):
            raise typer.BadParameter("Cannot change the evaluator of a frozen round")
        if not judge or judge_key != "judge" or write_as:
            raise typer.BadParameter("Round scoring uses its frozen judge and summary")
        judge_cfg = frozen.judge
        frozen_cases = rounds.directory(frozen.id) / "cases"
    summary = dataset.score_run(
        run_id, judge_cfg=judge_cfg, use_cache=cache, write_as=write_as, cases_dir=frozen_cases
    )
    solved = sum(c.resolved for c in summary.cases)
    where = summary_path(run_id).with_name(write_as) if write_as else summary_path(run_id)
    typer.echo(
        f"scored {len(summary.cases)} attempt(s): {solved} resolved; "
        f"judge={summary.judge or 'off'}; summary: {where}"
    )


@app.command()
def score(
    run: str = typer.Option(..., help="Run id under results/"),
    no_judge: bool = typer.Option(False, help="Skip the LLM code-quality judge"),
    no_cache: bool = typer.Option(False, help="Ignore DeepEval's metric cache"),
    judge_key: str = typer.Option(
        "judge", help="models.yaml entry to judge with; pick one that is not on the leaderboard"
    ),
    write_as: str | None = typer.Option(
        None, help="Write the summary under this file name instead of summary.json"
    ),
) -> None:
    """Score a run's attempt records with the DeepEval metrics and write summary.json."""
    _score(run, judge=not no_judge, cache=not no_cache, judge_key=judge_key, write_as=write_as)


@app.command()
def report(run: str = typer.Option(..., help="Run id under results/")) -> None:
    """Render summary.md (and refresh summary.json metrics) for a scored run."""
    from eval_harness.report.render import write_run_report
    from eval_harness.report.summary import load_summary

    js, md = write_run_report(load_summary(run))
    typer.echo(f"wrote {md} and {js}")


@app.command()
def reprice(
    run: str = typer.Option(..., help="Run id under results/, or a comma-separated list"),
) -> None:
    """Re-cost a finished run from its recorded tokens and the prices now in models.yaml.

    A record keeps the cost worked out when it ran, so a model that had no price then is
    still worth nothing afterwards. Token counts are recorded either way, so the cost can
    be derived again whenever a rate is known or changes.
    """
    from eval_harness.adapters.pricing import cost_usd
    from eval_harness.config import load_models
    from eval_harness.harness.record import load_record, save_record
    from eval_harness.report.render import write_run_report
    from eval_harness.report.summary import load_summary, summary_path

    models = load_models()
    for rid in [r.strip() for chunk in run.split(",") for r in [chunk] if r.strip()]:
        summary = load_summary(rid)
        prices = models[summary.model].price_per_mtok if summary.model in models else None
        if prices is None:
            typer.echo(f"{rid}: {summary.model} still has no price in models.yaml; skipped")
            continue
        before = after = 0.0
        for case in summary.cases:
            rec = load_record(rid, case.case_id)
            if rec is None:
                continue
            fresh = cost_usd(rec.usage, prices)
            before += rec.cost_usd
            after += fresh
            rec.cost_usd = fresh
            save_record(rec)
            case.cost_usd = fresh
        summary_path(rid).write_text(summary.model_dump_json(indent=2) + "\n")
        write_run_report(load_summary(rid))
        typer.echo(f"{rid} ({summary.model}): ${before:.2f} -> ${after:.2f}")


@app.command()
def init(
    repo: Path = typer.Option(  # noqa: B008
        ..., help="Path to the git repository to benchmark"
    ),
    key: str = typer.Option("", help="Short name for it; default the directory name"),
    out: Path | None = typer.Option(  # noqa: B008
        None, help="Where to write; default <data root>/config/repos.yaml"
    ),
    force: bool = typer.Option(False, help="Overwrite an existing file"),
) -> None:
    """Read a repository and write a first repos.yaml to correct."""
    from eval_harness.paths import data_root
    from eval_harness.scaffold import detect, render

    det = detect(repo, key)
    body = render(det)
    target = out or (data_root() / "config" / "repos.yaml")
    if target.exists() and not force:
        typer.echo(f"{target} exists; pass --force to overwrite, or --out to write elsewhere.")
        typer.echo("")
        typer.echo(body)
        raise typer.Exit(code=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    typer.echo(f"wrote {target}")

    # A repos.yaml alone is half a configuration; seed models.yaml from the shipped example so
    # the next command has something to run.
    models = target.parent / "models.yaml"
    if not models.exists():
        example = PROJECT_ROOT / "config" / "models.example.yaml"
        if example.exists():
            models.write_text(example.read_text())
            typer.echo(f"wrote {models} from the example — set one API key in it")
    tickets = f"tickets {det.ticket_prefix}" if det.ticket_prefix else "tickets from GitHub Issues"
    typer.echo(f"  {len(det.dep_dirs)} dependency root(s), {len(det.runners)} runner(s), {tickets}")
    for note in det.notes:
        typer.echo(f"  note: {note}")
    typer.echo("")
    typer.echo("Three things to check before anything else:")
    typer.echo("  1. the install command for each dep_dir, on a clean checkout")
    typer.echo("  2. the match globs — they must find your test files and nothing else")
    typer.echo("  3. the base image has whatever your suite needs beyond the language runtime")

    # Then say where they stand, rather than making them find out with another command.
    from eval_harness.doctor import run_all, worst
    from eval_harness.paths import reset_cache

    reset_cache()
    checks = run_all(online=True)
    typer.echo("")
    typer.echo("Where that leaves you:")
    for c in checks:
        if c.status != "ok":
            typer.echo(f"  {c.mark}  {c.name}: {c.detail}")
            if c.fix:
                typer.echo(f"          {c.fix}")
    typer.echo("")
    if worst(checks) == "fail":
        typer.echo("Fix the failures above, then: eval-harness doctor")
    else:
        typer.echo(
            "Ready. Next: eval-harness collect --repo "
            f"{det.key} --since YYYY-MM-DD --until YYYY-MM-DD --dry-run"
        )
        typer.echo("      or: eval-harness import swebench --limit 5")


@app.command(name="import")
def import_(
    dataset: str = typer.Argument(..., help="Only 'swebench' today"),
    variant: str = typer.Option("verified", help="verified | lite | full | multilingual"),
    limit: int = typer.Option(50, help="How many instances to import"),
    split: str = typer.Option("test", help="Dataset split"),
) -> None:
    """Import a public benchmark as cases, for calibration or a first run."""
    from eval_harness.collect.cases import save_case
    from eval_harness.collect.swebench import VARIANTS, environment_commit, fetch_rows, to_case
    from eval_harness.paths import cases_dir

    if dataset != "swebench":
        raise typer.BadParameter("only 'swebench' is supported today")
    if variant not in VARIANTS:
        raise typer.BadParameter(f"variant must be one of {', '.join(VARIANTS)}")

    rows = fetch_rows(variant, limit, split=split)
    repos: set[str] = set()
    envs = 0
    for row in rows:
        case = to_case(row, variant=variant)
        save_case(case, cases_dir())
        repos.add(case.repo)
        envs += bool(environment_commit(row))

    typer.echo(f"imported {len(rows)} instance(s) from {VARIANTS[variant]} into {cases_dir()}")
    typer.echo(
        f"  {len(repos)} repositor{'y' if len(repos) == 1 else 'ies'}: "
        f"{', '.join(sorted(repos)[:5])}{' …' if len(repos) > 5 else ''}"
    )
    typer.echo("")
    typer.echo("These are graded on named tests, not whole files, so they carry a criterion.")
    typer.echo("Before they can run you still need, per repository:")
    typer.echo("  - a repos.yaml entry with a local clone and a pytest runner")
    if envs:
        typer.echo(f"  - an environment pinned per instance ({envs} row(s) name one); repos.yaml")
        typer.echo("    keys dependency volumes by lockfile, which is not the same thing")
    typer.echo("")
    typer.echo("Then: eval-harness doctor")


@app.command()
def doctor(
    strict: bool = typer.Option(False, help="Exit non-zero on a warning, not only a failure"),
    offline: bool = typer.Option(
        False, help="Skip the round trips that check GitHub access and the Linear key"
    ),
) -> None:
    """Check everything a run depends on, before the run depends on it."""
    from eval_harness.doctor import run_all, worst

    checks = run_all(online=not offline)
    for c in checks:
        line = f"{c.mark}  {c.name:24} {c.detail}"
        typer.echo(line + (f"\n{'':6}{'':24} -> {c.fix}" if c.fix else ""))
    overall = worst(checks)
    typer.echo("")
    typer.echo({"ok": "ready", "warn": "usable, with warnings", "fail": "not ready"}[overall])
    if overall == "fail" or (strict and overall == "warn"):
        raise typer.Exit(code=1)


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Bind address; localhost by default"),
    port: int = typer.Option(8765),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open a browser tab"),
) -> None:
    """Serve the local dashboard: pick cases, launch runs, watch progress, read results."""
    from eval_harness.dashboard.app import serve

    typer.echo(f"dashboard on http://{host}:{port}/  (ctrl-c to stop)")
    serve(host=host, port=port, open_browser=open_browser)


@app.command()
def compare(
    runs: list[str] = typer.Option(  # noqa: B008
        ..., "--runs", help="Two or more run ids; the first is the baseline"
    ),
) -> None:
    """Head-to-head comparison with a regression list and paired bootstrap intervals."""
    from eval_harness.report.render import write_comparison
    from eval_harness.report.summary import load_summary

    ids = [r for chunk in runs for r in chunk.split(",") if r]
    if len(ids) < 2:
        raise typer.BadParameter("give at least two run ids")
    js, md = write_comparison([load_summary(r) for r in ids])
    typer.echo(f"wrote {md} and {js}")
