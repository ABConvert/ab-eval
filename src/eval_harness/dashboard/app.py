"""Starlette app: pick cases, launch harness jobs, watch progress, read results.

Everything it shows comes from files under data/ and results/; everything it starts is a
CLI subprocess through the single-slot queue in `jobs`.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from eval_harness.collect.cases import load_case
from eval_harness.config import Prices, load_models
from eval_harness.dashboard import config_io, data, jobs
from eval_harness.dashboard.security import DashboardBoundary
from eval_harness.paths import results_root
from eval_harness.report.render import case_tokens, compare_markdown, run_metrics

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))


def _fmt_secs(v: float | None) -> str:
    if v is None:
        return "—"
    v = float(v)
    return f"{v:.0f} s" if v < 90 else f"{v / 60:.1f} min"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.0f}%"


def _money(v: float | None) -> str:
    return "—" if v is None else f"${v:,.2f}"


def _thousands(v: float | int | None) -> str:
    """Compact token counts: 15,122,321 -> 15.1M, 740,559 -> 741k."""
    if v is None:
        return "—"
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


def _diff_lines(diff: str | None) -> list[dict[str, Any]]:
    """Split a unified diff into classed lines a template can render.

    A diff was being printed as one escaped blob inside `pre.log`, which wraps: a long
    removed line broke across two rows and its remainder began in the column an
    unchanged line begins in, so the one-line-one-change contract that makes a diff
    readable was gone. Splitting it here is what lets the markup carry the sign in a
    fixed gutter, give the line a class, and keep real line numbers.

    `kind` is one of file / hunk / add / del / ctx / meta. `old` and `new` are the line
    numbers in each side, or None where that side has no line.
    """
    out: list[dict[str, Any]] = []
    if not diff:
        return out
    old = new = 0
    for raw in diff.splitlines():
        if raw.startswith("diff --git ") or raw.startswith(("--- ", "+++ ")):
            out.append({"kind": "file", "text": raw, "old": None, "new": None})
        elif raw.startswith("@@"):
            # @@ -142,7 +142,9 @@ context — the two starts are where the numbering resumes.
            old, new = 0, 0
            head = raw.split("@@")[1] if "@@" in raw[2:] else ""
            for part in head.split():
                if part.startswith("-"):
                    old = int(part[1:].split(",")[0] or 0)
                elif part.startswith("+"):
                    new = int(part[1:].split(",")[0] or 0)
            out.append({"kind": "hunk", "text": raw, "old": None, "new": None})
        elif raw.startswith("+"):
            out.append({"kind": "add", "text": raw[1:], "old": None, "new": new})
            new += 1
        elif raw.startswith("-"):
            out.append({"kind": "del", "text": raw[1:], "old": old, "new": None})
            old += 1
        elif raw.startswith("\\"):  # "\ No newline at end of file"
            out.append({"kind": "meta", "text": raw, "old": None, "new": None})
        else:
            out.append(
                {
                    "kind": "ctx",
                    "text": raw[1:] if raw.startswith(" ") else raw,
                    "old": old,
                    "new": new,
                }
            )
            old += 1
            new += 1
    return out


def _diff_stat(diff: str | None) -> dict[str, int]:
    """Added and removed line counts, so a diff can say its size before you read it."""
    lines = _diff_lines(diff)
    return {
        "add": sum(1 for row in lines if row["kind"] == "add"),
        "del": sum(1 for row in lines if row["kind"] == "del"),
        "files": sum(1 for row in lines if row["text"].startswith("diff --git ")),
    }


templates.env.filters["tokens"] = case_tokens
templates.env.filters["compact"] = _thousands
templates.env.filters["secs"] = _fmt_secs
templates.env.filters["pct"] = _pct
templates.env.filters["money"] = _money
templates.env.filters["difflines"] = _diff_lines
templates.env.filters["diffstat"] = _diff_stat


def _ctx(request: Request, **kw: Any) -> dict[str, Any]:
    q = jobs.queue()
    return {
        "request": request,
        "current_job": q.current(),
        "queued_jobs": q.queued(),
        "now": datetime.now(UTC).strftime("%H:%M:%S"),
        # ISO so a template can do arithmetic against started_at; "now" is display-only.
        "now_iso": datetime.now(UTC).isoformat(timespec="seconds"),
        **kw,
    }


def _missing(
    request: Request,
    *,
    heading: str,
    detail: str,
    wanted: str = "",
    back_href: str = "/runs",
    back_label: str = "All runs",
    page: str = "runs",
) -> HTMLResponse:
    """A 404 that is still this product: the shell, what was asked for, and a way out."""
    return templates.TemplateResponse(
        request,
        "missing.html",
        _ctx(
            request,
            heading=heading,
            detail=detail,
            wanted=wanted,
            back_href=back_href,
            back_label=back_label,
            page=page,
        ),
        status_code=404,
    )


async def cases_page(request: Request) -> HTMLResponse:
    rows = data.case_rows()
    # `eval-harness dashboard` opens the browser here, and models.yaml is gitignored, so
    # on a clone of this repo it does not exist yet. This call was unguarded, which made
    # the first page a stranger sees a bare Internal Server Error with no nav and no way
    # on — while /setup, the page built to tell them exactly this, sat one URL away and
    # unreachable. An install that is not configured is a state to render, not a crash.
    try:
        runnable = [k for k in load_models() if k not in ("judge", "classifier")]
    except (FileNotFoundError, ValueError):
        runnable = []
    counts = {
        "all": len(rows),
        "dev": sum(1 for r in rows if r.split == "dev"),
        "holdout": sum(1 for r in rows if r.split == "holdout"),
        "bug_fix": sum(1 for r in rows if r.kind == "bug_fix"),
        "feature": sum(1 for r in rows if r.kind == "feature"),
        "valid": sum(1 for r in rows if r.validation == "valid"),
        "invalid": sum(1 for r in rows if r.validation == "invalid"),
        "pending": sum(1 for r in rows if r.validation in ("pending", "error")),
        "easy": sum(1 for r in rows if r.difficulty == "easy"),
        "medium": sum(1 for r in rows if r.difficulty == "medium"),
        "hard": sum(1 for r in rows if r.difficulty == "hard"),
    }
    return templates.TemplateResponse(
        request,
        "cases.html",
        _ctx(
            request,
            rows=rows,
            counts=counts,
            # Two lists, because the picker always offers the reference and the page has
            # to be able to say when that is the ONLY thing it can offer. One list with
            # human-patch prepended can never be empty, so the note that explains an
            # empty one could not fire — computed, and then not reachable.
            models=["human-patch", *runnable],
            runnable=runnable,
            default_run_id=f"run-{datetime.now(UTC).strftime('%m%d-%H%M')}",
            page="cases",
        ),
    )


def _opt_int(raw: Any) -> int | None:
    """A blank cap box means "whatever this model declares", not a number picked here."""
    text = str(raw or "").strip()
    return int(text) if text else None


async def launch(request: Request) -> RedirectResponse:
    form = await request.form()
    action = str(form.get("action") or "run")
    cases = [c for c in form.getlist("cases") if isinstance(c, str)]
    if not cases and action in ("run", "validate"):
        return RedirectResponse("/?error=pick+at+least+one+case", status_code=303)
    q = jobs.queue()
    if action == "validate":
        job = q.submit(jobs.validate_job(cases))
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)
    model = str(form.get("model") or "claude-opus-5")
    run_id = str(form.get("run_id") or "").strip() or f"run-{datetime.now(UTC):%m%d-%H%M}"
    job = q.submit(
        jobs.run_job(
            model=model,
            run_id=run_id,
            cases=cases,
            max_turns=_opt_int(form.get("max_turns")),
            wall_clock=_opt_int(form.get("wall_clock")),
        )
    )
    return RedirectResponse(f"/runs/{job.run_id}", status_code=303)


async def runs_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "runs.html",
        _ctx(
            request,
            runs=data.runs(),
            run_datasets={r.run_id: data.dataset_of(r) for r in data.runs()},
            comparisons=data.comparison_files(),
            history=jobs.queue().recent(),
            page="runs",
        ),
    )


def _run_or_404(run_id: str) -> data.RunView:
    view = data.load_run(run_id)
    if view is None:
        raise FileNotFoundError(run_id)
    return view


async def run_page(request: Request) -> HTMLResponse:
    run_id = request.path_params["run_id"]
    try:
        view = _run_or_404(run_id)
    except FileNotFoundError:
        return _missing(
            request,
            heading="No run by that name",
            detail=(
                "Runs are folders under results/, and this one is not there. It may have been "
                "renamed or deleted, or the link may be from a saved comparison file that "
                "outlived it."
            ),
            wanted=run_id,
        )
    job = jobs.queue().active_for_run(run_id)
    # A run is live while a job is attached, and otherwise only while it is still
    # unfinished AND unscored. Records can be missing for reasons that have nothing
    # to do with progress — pruned from disk, or a case that crashed without writing
    # one — and reading that as "still going" left eight of twenty-one runs, every
    # one the leaderboard ranks, polling a dead job and unable to reach their own
    # results. A scored run is finished by definition, however many records survive.
    live = job is not None or (view.done < view.total and view.summary is None)
    if live:
        return templates.TemplateResponse(
            request,
            "run_progress.html",
            _ctx(request, view=view, job=job, dataset=data.dataset_of(view), page="runs"),
        )
    return templates.TemplateResponse(
        request,
        "run_results.html",
        _ctx(
            request,
            view=view,
            dataset=data.dataset_of(view),
            metrics=run_metrics(view.summary) if view.summary else None,
            all_runs=[r.run_id for r in data.runs() if r.run_id != run_id],
            page="runs",
        ),
    )


async def run_panel(request: Request) -> HTMLResponse:
    """HTMX partial: the live table plus the log tail, polled while a run is active."""
    run_id = request.path_params["run_id"]
    try:
        view = _run_or_404(run_id)
    except FileNotFoundError:
        return HTMLResponse("", status_code=404)
    job = jobs.queue().active_for_run(run_id)
    # This partial is only armed while a job is attached, so a poll arriving with no
    # job means the job ended between one tick and the next — and the page around this
    # panel was rendered for a live run, so it is now wrong whichever way the run went.
    # Hand back to the router: htmx reloads, run_page re-decides, and the reloaded page
    # arms nothing unless a job is attached again, so this cannot re-arm itself.
    # Keyed on the job alone, not on pending_ids: a run whose records were pruned still
    # has 104 pending ids after a re-score, and reading that as "not finished yet" is
    # what stranded every ranked run before e4eb0bc.
    settled = job is None
    headers = {"HX-Refresh": "true"} if settled else None
    return templates.TemplateResponse(
        request, "_run_panel.html", _ctx(request, view=view, job=job), headers=headers
    )


async def case_page(request: Request) -> HTMLResponse:
    run_id, case_id = request.path_params["run_id"], request.path_params["case_id"]
    records = data.load_records(run_id)
    rec = records.get(case_id)
    if rec is None:
        return _missing(
            request,
            heading="That run did not attempt this case",
            detail=(
                "The run exists but has no record for this case — it was not in the set, or the "
                "run stopped before reaching it, or the record was pruned."
            ),
            wanted=f"{case_id} in {run_id}",
            back_href=f"/runs/{quote(run_id)}",
            back_label="Back to the run",
        )
    case = load_case(case_id)
    score = None
    view = data.load_run(run_id)
    if view and view.summary:
        score = next((c for c in view.summary.cases if c.case_id == case_id), None)
    return templates.TemplateResponse(
        request,
        "case.html",
        _ctx(
            request,
            rec=rec,
            case=case,
            title=data.title_of(case),
            score=score,
            # Already loaded above for the score. The judge's name and the date it ran
            # live on the summary, not on the case score, and a judgement with neither
            # is an opinion from nobody at no particular time.
            summary=view.summary if view else None,
            series_order=_series_order(),
            run_id=run_id,
            page="runs",
        ),
    )


async def case_across_page(request: Request) -> HTMLResponse:
    """One case, every model that attempted it."""
    case_id = request.path_params["case_id"]
    try:
        case = load_case(case_id)
    except FileNotFoundError:
        return _missing(
            request,
            heading="No case by that id",
            detail=(
                "Cases live as JSON under data/cases/. This one has not been curated, or the id "
                "is mistyped — they look like ENG-1234."
            ),
            wanted=case_id,
            back_href="/",
            back_label="All cases",
            page="cases",
        )
    from eval_harness.collect import datasets as ds_mod

    attempts = data.case_across_runs(case_id)
    return templates.TemplateResponse(
        request,
        "case_across.html",
        _ctx(
            request,
            case=case,
            title=data.title_of(case),
            attempts=attempts,
            # Which sets this case belongs to. A case nobody has run yet needs to say
            # where it can be found, and "it is curated and ready" is only useful if
            # the page says which set to look in.
            in_datasets=ds_mod.containing(case_id),
            series_order=_series_order(),
            page="cases",
        ),
    )


async def datasets_page(request: Request) -> HTMLResponse:
    from eval_harness.collect import datasets as ds_mod
    from eval_harness.metrics.test_quality import load_review

    picked = request.query_params.get("name") or ""
    current = None
    rows: list[dict[str, Any]] = []
    if picked:
        try:
            current = ds_mod.load(picked)
        except FileNotFoundError:
            current = None
    all_rows = data.case_rows()
    if current is not None:
        by_id = {r.case_id: r for r in all_rows}
        rows = [
            {"row": by_id.get(cid), "case_id": cid, "review": load_review(cid)}
            for cid in current.case_ids
        ]
    return templates.TemplateResponse(
        request,
        "datasets.html",
        _ctx(
            request,
            datasets=ds_mod.load_all(),
            current=current,
            rows=rows,
            all_cases=all_rows,
            error=request.query_params.get("error"),
            page="datasets",
        ),
    )


async def dataset_create(request: Request) -> RedirectResponse:
    from eval_harness.collect import datasets as ds_mod

    form = await request.form()
    try:
        ds = ds_mod.create(
            str(form.get("name") or ""),
            str(form.get("description") or ""),
            [c for c in form.getlist("cases") if isinstance(c, str)],
        )
    except (ValueError, FileExistsError) as e:
        return RedirectResponse(f"/datasets?error={quote(str(e))}", status_code=303)
    return RedirectResponse(f"/datasets?name={ds.name}", status_code=303)


async def dataset_edit(request: Request) -> RedirectResponse:
    """Add cases, add a Linear ticket (collected as a job first), or remove cases."""
    from eval_harness.collect import datasets as ds_mod

    name = request.path_params["name"]
    form = await request.form()
    action = str(form.get("action") or "add")
    ticket = str(form.get("ticket") or "").strip()
    ids = [c for c in form.getlist("cases") if isinstance(c, str)]
    try:
        if ticket:
            job = jobs.queue().submit(
                jobs.add_ticket_job(name, ticket.upper(), str(form.get("kind") or "bug_fix"))
            )
            return RedirectResponse(f"/jobs/{job.id}", status_code=303)
        if action == "remove":
            ds_mod.remove_cases(name, ids)
        elif ids:
            ds_mod.add_cases(name, ids)
    except (ValueError, LookupError, FileNotFoundError) as e:
        return RedirectResponse(f"/datasets?name={name}&error={quote(str(e))}", status_code=303)
    return RedirectResponse(f"/datasets?name={name}", status_code=303)


async def dataset_review(request: Request) -> RedirectResponse:
    """Queue the test-quality judge over selected cases, or the whole dataset."""
    from eval_harness.collect import datasets as ds_mod

    name = request.path_params["name"]
    form = await request.form()
    ids = [c for c in form.getlist("cases") if isinstance(c, str)]
    if not ids:
        try:
            ids = ds_mod.load(name).case_ids
        except FileNotFoundError:
            ids = []
    if not ids:
        return RedirectResponse(f"/datasets?name={name}", status_code=303)
    job = jobs.queue().submit(jobs.review_tests_job(ids))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


def _editable(request: Request) -> bool:
    """Whether the local dashboard offers its configuration forms."""
    return bool(getattr(request.app.state, "can_write", False))


def _repo_shallow(entry: dict[str, Any]) -> dict[str, Any]:
    """The fields of one repository the form covers, read off the raw mapping."""
    limits = entry.get("limits") or {}
    if not isinstance(limits, dict):
        limits = {}
    authors = entry.get("agent_authors") or {}
    if not isinstance(authors, dict):
        authors = {}
    return {
        "github": entry.get("github", ""),
        "local_path": str(entry.get("local_path", "")),
        "linear_team": entry.get("linear_team", ""),
        "image": entry.get("image", ""),
        "dockerfile": entry.get("dockerfile", ""),
        "cpus": limits.get("cpus", 2),
        "memory": limits.get("memory", "6g"),
        "pids": limits.get("pids", 2048),
        "agent_authors": "\n".join(f"{k}: {v}" for k, v in authors.items()),
        # What the form does not reach, counted, so the page can say what it is leaving alone
        # rather than leaving the reader to wonder whether saving will drop it.
        "dep_dirs": len(entry.get("dep_dirs") or []),
        "runners": list((entry.get("runners") or {}).keys()),
        "test_file_globs": len(entry.get("test_file_globs") or []),
        "non_code_globs": len(entry.get("non_code_globs") or []),
    }


async def _setup_render(
    request: Request,
    *,
    error: str = "",
    error_for: str = "",
    drafts: dict[str, str] | None = None,
    form_draft: dict[str, Any] | None = None,
    saved: str = "",
    status: int = 200,
) -> HTMLResponse:
    """Render /setup. One path, whether it was reached by a GET or by a save that failed.

    `drafts` (YAML text) and `form_draft` (one entry's fields) are what the reader typed, and
    a rejected save has to hand them back. A form that reloads itself from disk on an error
    makes them retype eight fields to fix the one they got wrong, which is how the second
    mistake gets made. `form_draft` also says which entry to open, so one bad save expands
    the form it came from rather than every form on the page.
    """
    from eval_harness.dashboard import config_io
    from eval_harness.doctor import run_all, worst
    from eval_harness.paths import data_root

    # run_all shells out to docker and the provider CLIs. Calling it inline would block the
    # event loop for as long as those take, freezing every other page in the dashboard.
    checks = await asyncio.to_thread(run_all)
    drafts = drafts or {}

    configs: list[dict[str, Any]] = []
    entries: dict[str, dict[str, Any]] = {}
    for name in ("repos.yaml", "models.yaml"):
        path = config_io.read_path(name)
        target = config_io.write_path(name)
        try:
            body = path.read_text()
        except OSError:
            body = ""
        try:
            raw = config_io.load_raw(name)
            parse_error = ""
        except config_io.ConfigError as e:
            raw, parse_error = {}, str(e)
        backups = config_io.backups(name)
        entries[name] = raw
        configs.append(
            {
                "name": name,
                "path": str(path),
                "write_path": str(target),
                # A data root without its own config reads the checkout's. Saving then writes a
                # new file that takes precedence over it, which is worth saying before it happens
                # rather than leaving someone to wonder why their edit did not seem to apply.
                "shadows": path.exists() and path != target,
                "body": body,
                "draft": drafts.get(name, ""),
                "exists": path.exists(),
                "entries": raw,
                "parse_error": parse_error,
                "backup": backups[0].name if backups else "",
                "backup_count": len(backups),
            }
        )

    models_cfg = next(c for c in configs if c["name"] == "models.yaml")
    repos = {k: _repo_shallow(v) for k, v in entries["repos.yaml"].items() if isinstance(v, dict)}

    # Which variables are set, as booleans only. The value is never read into the page.
    env_set = {
        str(m["api_key_env"]): bool(os.environ.get(str(m["api_key_env"])))
        for m in entries["models.yaml"].values()
        if isinstance(m, dict) and m.get("api_key_env")
    }

    return templates.TemplateResponse(
        request,
        "setup.html",
        _ctx(
            request,
            checks=checks,
            overall=worst(checks),
            configs=configs,
            models=models_cfg,
            repos=repos,
            env_set=env_set,
            providers=config_io.PROVIDERS,
            provider_fields={k: list(v) for k, v in config_io.PROVIDER_FIELDS.items()},
            gated_fields=list(config_io.GATED_FIELDS),
            provider_fields_json=json.dumps(
                {
                    **{k: list(v) for k, v in config_io.PROVIDER_FIELDS.items()},
                    "__all__": list(config_io.GATED_FIELDS),
                }
            ),
            special_keys=config_io.SPECIAL_KEYS,
            can_write=_editable(request),
            bind_host=getattr(request.app.state, "bind_host", "127.0.0.1"),
            error=error,
            error_for=error_for,
            form_draft=form_draft,
            saved=saved,
            data_root=str(data_root()),
            page="setup",
        ),
        status_code=status,
    )


async def setup_page(request: Request) -> HTMLResponse:
    """What is configured, what is missing, and — on loopback — the forms that fix it."""
    return await _setup_render(request, saved=request.query_params.get("saved", ""))


async def _saved(request: Request, name: str, backup: Path | None, anchor: str) -> Response:
    """Redirect back to the page that asked, which then re-runs the checks the save changed."""
    note = f"{name} saved" + (f", previous version kept as {backup.name}" if backup else "")
    return RedirectResponse(f"/setup?saved={quote(note)}#{anchor}", status_code=303)


# What the repository form posts, so a refused save can hand back exactly what was in it.
_REPO_FIELDS = (
    "github",
    "local_path",
    "linear_team",
    "image",
    "dockerfile",
    "cpus",
    "memory",
    "pids",
    "agent_authors",
)


def _typed_prices(form: Any) -> dict[str, Any]:
    """The four price boxes as they were typed, for redisplay after a refusal."""
    typed = {p: str(form.get(f"price_{p}", "")).strip() for p in Prices.model_fields}
    return {"price_per_mtok": {p: v for p, v in typed.items() if v}} if any(typed.values()) else {}


def _num(form: Any, key: str) -> float | None:
    raw = str(form.get(key, "")).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as e:
        raise config_io.ConfigError(f"{key} must be a number; got {raw!r}.") from e


# The four run budgets, as `caps_<field>` on the form. Blank means the field is absent from
# the block, which means the harness default — so a model declares only the cap it needs to
# be different, and the rest keep moving with the defaults.
_CAP_FIELDS = {
    "max_turns": "turns",
    "wall_clock_seconds": "seconds",
    "max_output_tokens_total": "tokens",
    "tool_timeout_seconds": "seconds",
}


def _typed_caps(form: Any) -> dict[str, Any]:
    """The cap boxes as they were typed, for redisplay after a refusal."""
    typed = {c: str(form.get(f"caps_{c}", "")).strip() for c in _CAP_FIELDS}
    return {"caps": {c: v for c, v in typed.items() if v}} if any(typed.values()) else {}


def _caps_block(form: Any) -> dict[str, Any]:
    """`caps:` as integers, or nothing at all when every box was left empty."""
    given: dict[str, Any] = {}
    for cap, unit in _CAP_FIELDS.items():
        value = _num(form, f"caps_{cap}")
        if value is None:
            continue
        if value != int(value) or int(value) < 1:
            raise config_io.ConfigError(
                f"caps.{cap} is a whole number of {unit} and must be at least 1; got {value:g}."
            )
        given[cap] = int(value)
    return {"caps": given} if given else {}


async def model_save(request: Request) -> Response:
    """Add or update one entry in models.yaml. Creates the file if it is not there yet."""

    form = await request.form()
    opened_on = request.path_params.get("key", "")
    key = str(form.get("key", "")).strip()
    body: dict[str, Any] = {}
    try:
        if not key:
            raise config_io.ConfigError("Every entry needs a key — it is how you pick the model.")
        provider = str(form.get("provider", "")).strip()
        # A field the chosen provider cannot send is not read off the form even when the
        # browser posted it — with JavaScript off every box is visible, and a temperature
        # typed against claude-code would otherwise be written and then refused by the
        # loader. What the entry already had is carried over by `keep` below.
        usable = config_io.fields_for(provider)
        body = {
            "provider": provider,
            "model": str(form.get("model", "")).strip(),
            "effort": str(form.get("effort", "")).strip() or "high",
        }
        if "max_output_tokens" in usable:
            body["max_output_tokens"] = int(_num(form, "max_output_tokens") or 16000)
        if "temperature" in usable:
            temperature = _num(form, "temperature")
            if temperature is not None:
                body["temperature"] = temperature
        if "reasoning_param" in usable:
            param = str(form.get("reasoning_param", "")).strip()
            if param and param != "omit":
                body["reasoning_param"] = param
        prices = {
            p: _num(form, f"price_{p}") for p in ("input", "output", "cache_read", "cache_write")
        }
        given = {p: v for p, v in prices.items() if v is not None}
        if given and len(given) < 4:
            missing = ", ".join(p for p, v in prices.items() if v is None)
            raise config_io.ConfigError(
                "Prices are all four or none: the harness bills cached reads and writes at "
                f"their own rates, and a run that knew only some of them would report a cost "
                f"that is quietly wrong. Still empty: {missing}."
            )
        if given:
            body["price_per_mtok"] = given
        for optional in ("base_url", "api_key_env"):
            value = str(form.get(optional, "")).strip()
            if value:
                body[optional] = value
        body.update(_caps_block(form))
        backup = config_io.put_model(
            key,
            body,
            replacing=opened_on,
            keep=tuple(f for f in config_io.GATED_FIELDS if f not in usable),
        )
    except config_io.ConfigError as e:
        return await _setup_render(
            request,
            error=str(e),
            error_for="models.yaml",
            # The prices are read back out of `body` by the form, so a partial set has to
            # survive as what was typed rather than as the four keys pydantic wanted.
            form_draft={
                "file": "models.yaml",
                "opened_on": opened_on,
                "key": key,
                "body": config_io.redact({**body, **_typed_prices(form), **_typed_caps(form)}),
            },
            status=422,
        )
    return await _saved(request, "models.yaml", backup, "models")


async def model_delete(request: Request) -> Response:
    try:
        backup = config_io.drop_model(request.path_params["key"])
    except config_io.ConfigError as e:
        return await _setup_render(request, error=str(e), error_for="models.yaml", status=422)
    return await _saved(request, "models.yaml", backup, "models")


async def models_from_example(request: Request) -> Response:
    """Create models.yaml from the example the harness ships, comments and all.

    The one write here that does not re-serialise: the example's text goes through unchanged,
    because its value is the commented walk through four kinds of provider.
    """
    from eval_harness import PROJECT_ROOT

    example = PROJECT_ROOT / "config" / "models.example.yaml"
    try:
        if not example.exists():
            raise config_io.ConfigError(
                f"This checkout has no {example}. Add a model with the form instead."
            )
        backup = config_io.save("models.yaml", example.read_text())
    except config_io.ConfigError as e:
        return await _setup_render(request, error=str(e), error_for="models.yaml", status=422)
    return await _saved(request, "models.yaml", backup, "models")


async def repo_save(request: Request) -> Response:
    """Save the shallow fields of one repository, leaving everything nested as it was."""

    form = await request.form()
    key = request.path_params["key"]
    try:
        authors: dict[str, str] = {}
        for line in str(form.get("agent_authors", "")).splitlines():
            line = line.strip()
            if not line:
                continue
            login, _, kind = line.partition(":")
            kind = kind.strip() or "agent"
            if kind not in ("agent", "unclear"):
                raise config_io.ConfigError(
                    f"agent_authors: {kind!r} is not a kind. Each line is `login: agent` or "
                    "`login: unclear` — whether a human wrote the pull request is what the "
                    "benchmark is comparing against."
                )
            authors[login.strip()] = kind
        fields: dict[str, Any] = {
            "github": str(form.get("github", "")).strip(),
            "local_path": str(form.get("local_path", "")).strip(),
            "linear_team": str(form.get("linear_team", "")).strip(),
            "image": str(form.get("image", "")).strip(),
            "dockerfile": str(form.get("dockerfile", "")).strip(),
            "agent_authors": authors,
            "limits": {
                "cpus": config_io.tidy_number(_num(form, "cpus") or 2),
                "memory": str(form.get("memory", "")).strip() or "6g",
                "pids": int(_num(form, "pids") or 2048),
            },
        }
        backup = config_io.patch_repo(key, fields)
    except config_io.ConfigError as e:
        return await _setup_render(
            request,
            error=str(e),
            error_for="repos.yaml",
            form_draft={
                "file": "repos.yaml",
                "opened_on": key,
                "key": key,
                "body": config_io.redact({k: str(form.get(k, "")) for k in _REPO_FIELDS}),
            },
            status=422,
        )
    return await _saved(request, "repos.yaml", backup, "repos")


async def repo_detect(request: Request) -> Response:
    """Read a repository and draft a repos.yaml for it. Writes nothing.

    Exactly what `eval-harness init` does, and for the same reason it is a draft rather than a
    save: every value it finds is a guess about someone else's test setup, and the CHECK markers
    are there to be read before the file is real.
    """

    form = await request.form()
    raw = str(form.get("path", "")).strip()
    key = str(form.get("key", "")).strip()
    try:
        if not raw:
            raise config_io.ConfigError("Give the path to a git checkout on this machine.")
        repo = Path(raw).expanduser()
        if not repo.is_dir():
            raise config_io.ConfigError(f"{repo} is not a directory on this machine.")
        if not (repo / ".git").is_dir():
            raise config_io.ConfigError(
                f"{repo} has no .git in it. A case is a merged pull request replayed against "
                "the commit before it, so this has to be the checkout itself."
            )
        from eval_harness.scaffold import detect, render

        # Walks the tree and shells out to git; off the event loop like the doctor's checks.
        detection = await asyncio.to_thread(detect, repo, key)
        draft = render(detection)
    except config_io.ConfigError as e:
        return await _setup_render(request, error=str(e), error_for="repos.yaml", status=422)

    existing = config_io.read_path("repos.yaml")
    if existing.exists():
        # Below what is there, not over it: this drafts one repository and the file may hold
        # others. The editor's own validation catches the name collision if there is one.
        draft = existing.read_text().rstrip("\n") + "\n\n" + draft
    return await _setup_render(
        request,
        drafts={"repos.yaml": draft},
        saved=f"Drafted {detection.key} from {repo} — nothing written yet. Read the CHECK "
        "lines, then save.",
    )


async def config_raw_save(request: Request) -> Response:
    """Save one whole file exactly as typed. The escape hatch, and the only lossless path."""

    name = request.path_params["name"]
    form = await request.form()
    text = str(form.get("body", ""))
    try:
        if name not in config_io.FILES:
            raise config_io.ConfigError(f"{name} is not a file this page writes.")
        backup = config_io.save(name, text)
    except config_io.ConfigError as e:
        # Everything typed comes back, except a credential — echoing that into the textarea
        # would put the key on the page, in the browser history and in any screenshot of it,
        # which is exactly what refusing it was for. They still have their own paste.
        keep = config_io.find_secret(text) is None
        return await _setup_render(
            request,
            error=str(e),
            error_for=name,
            drafts={name: text} if keep else None,
            status=422,
        )
    return await _saved(request, name, backup, "repos" if name == "repos.yaml" else "models")


async def leaderboard_page(request: Request) -> HTMLResponse:
    from eval_harness.report.leaderboard import WEIGHTS, build

    scored = [r for r in data.runs() if r.scored]
    labels = {r.run_id: data.dataset_of(r) for r in scored}
    # Default to the dataset with the most scored runs rather than mixing them: a run over
    # one case would otherwise outrank a run over fourteen on the same resolve rate.
    counts: dict[str, int] = {}
    for d in labels.values():
        if d:
            counts[d] = counts.get(d, 0) + 1
    # Most runs wins; ties go to the dataset with more cases, because the larger benchmark is
    # the one worth landing on.
    sizes = {d: len(data.dataset_case_ids(d)) for d in counts}
    busiest = max(counts, key=lambda k: (counts[k], sizes.get(k, 0), k)) if counts else ""
    requested = request.query_params.get("dataset")
    wanted = busiest if requested is None else requested
    if wanted:
        scored = [r for r in scored if labels.get(r.run_id) == wanted]
    board = (
        build([r.summary for r in scored if r.summary])
        if scored
        else {"weights": WEIGHTS, "rows": []}
    )
    return templates.TemplateResponse(
        request,
        "leaderboard.html",
        _ctx(
            request,
            board=board,
            datasets=sorted({d for d in labels.values() if d}),
            mixed_warning=not wanted and len({r.total for r in scored}) > 1,
            wanted=wanted,
            run_datasets=labels,
            headline=_leaderboard_headline(wanted, scored),
            series_order=_series_order(),
            page="leaderboard",
        ),
    )


def _leaderboard_headline(wanted: str, scored: list[data.RunView]) -> dict[str, object]:
    """One line of context so a screenshot of this page explains itself."""
    from eval_harness.collect.datasets import load as load_dataset

    description = ""
    cases = 0
    try:
        ds = load_dataset(wanted) if wanted else None
        if ds:
            description, cases = ds.description, len(ds)
    except Exception:
        pass
    attempts = sum(r.total for r in scored)
    return {
        "models": len({r.model for r in scored}),
        "runs": len(scored),
        "cases": cases,
        "attempts": attempts,
        "description": description,
    }


def _series_order() -> list[str]:
    """One ordering of every model in the data, so a mark means the same thing anywhere.

    Sorting the models on the page instead puts a model at a different index in every
    selection: opus-5 is the second series beside fable and the first beside sonnet, so
    it changes colour and shape when you change which runs you are comparing — which is
    the defect the hard-coded PALETTE array had, arrived at a different way. series.js
    takes this same list, so the canvas and the DOM cannot drift apart either.
    """
    return sorted({r.model for r in data.runs() if r.model})


def _compare_paths(run_ids: list[str]) -> list[str]:
    """The two files `Write compare files` overwrites, named before it is pressed.

    Mirrors report.render.write_comparison, which builds the name from every run id.
    A mutating control that does not say what it writes is asking for trust it has not
    earned — and at four runs this is a 78-character filename in results/, beside the
    run folders, not somewhere out of the way.
    """
    name = "compare-" + "-vs-".join(run_ids)
    root = results_root()
    return [str(root / f"{name}.{ext}") for ext in ("md", "json")]


async def compare_page(request: Request) -> HTMLResponse:
    picked = request.query_params.getlist("runs")
    all_runs = data.runs()
    # Each run's size travels with its name. Comparing a 104-case run with a 14-case one
    # is the mistake this page most easily lets you make, and the picker was a row of
    # bare ids with nothing to tell them apart.
    available = [
        {
            "run_id": r.run_id,
            "model": r.model,
            "total": r.total,
            "dataset": data.dataset_of(r),
        }
        for r in all_runs
        if r.scored
    ]
    # Show unscored runs too, greyed out: silently omitting them looks like they are missing.
    unscored = [
        {
            "run_id": r.run_id,
            "model": r.model,
            "done": r.done,
            "total": r.total,
            "resolved": r.resolved,
        }
        for r in all_runs
        if not r.scored and r.done
    ]
    payload, summaries = None, []
    by_run: dict[str, dict[str, Any]] = {}
    case_ids: list[str] = []
    if len(picked) >= 2:
        summaries = data.summaries(picked)
        if len(summaries) >= 2:
            _, payload = compare_markdown(summaries)
            by_run = {s.run_id: {c.case_id: c for c in s.cases} for s in summaries}
            # Every case any run attempted, not just the shared ones: a run that did not
            # attempt a case is a fact about that run, and the grid says "No attempt"
            # rather than leaving the row out. The paired statistics above are computed
            # over the intersection either way — payload.common_cases says which set.
            every = {cid for cells in by_run.values() for cid in cells}

            # Disagreements first, because that is what the sub-heading promises and it
            # is the only part of a 104-row grid worth reading top to bottom.
            def _agreed(cid: str) -> bool:
                got = [
                    bool(by_run[s.run_id].get(cid) and by_run[s.run_id][cid].resolved)
                    for s in summaries
                ]
                return all(got) or not any(got)

            case_ids = sorted(every, key=lambda cid: (_agreed(cid), cid))
    return templates.TemplateResponse(
        request,
        "compare.html",
        _ctx(
            request,
            available=available,
            unscored=unscored,
            picked=picked,
            payload=payload,
            summaries=summaries,
            by_run=by_run,
            case_ids=case_ids,
            series_order=_series_order(),
            compare_paths=_compare_paths(picked) if len(picked) >= 2 else None,
            page="compare",
        ),
    )


async def compare_launch(request: Request) -> RedirectResponse:
    form = await request.form()
    picked = [r for r in form.getlist("runs") if isinstance(r, str)]
    if len(picked) < 2:
        return RedirectResponse("/compare", status_code=303)
    jobs.queue().submit(jobs.compare_job(picked))
    return RedirectResponse("/compare?" + "&".join(f"runs={r}" for r in picked), status_code=303)


async def score_run(request: Request) -> RedirectResponse:
    run_id = request.path_params["run_id"]
    jobs.queue().submit(jobs.score_job(run_id))
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


async def job_page(request: Request) -> HTMLResponse:
    job = jobs.queue().get(request.path_params["job_id"])
    if job is None:
        return _missing(
            request,
            heading="No job by that id",
            detail=(
                "The queue keeps jobs for this session and a while after. This one has been "
                "cleared, or it was queued by a different dashboard process."
            ),
            wanted=request.path_params["job_id"],
            back_href="/runs#jobs",
            back_label="Recent jobs",
        )
    return templates.TemplateResponse(request, "job.html", _ctx(request, job=job, page="runs"))


async def job_log(request: Request) -> HTMLResponse:
    job = jobs.queue().get(request.path_params["job_id"])
    if job is None:
        return HTMLResponse("", status_code=404)
    return templates.TemplateResponse(request, "_job_log.html", _ctx(request, job=job))


async def job_cancel(request: Request) -> RedirectResponse:
    job_id = request.path_params["job_id"]
    jobs.queue().cancel(job_id)
    referer = request.headers.get("referer") or "/"
    return RedirectResponse(referer, status_code=303)


async def queue_strip(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "_queue.html", _ctx(request))


async def api_state(request: Request) -> JSONResponse:
    q = jobs.queue()
    current = q.current()
    return JSONResponse(
        {
            "current": jobs.job_dict(current) if current else None,
            "queued": [jobs.job_dict(j) for j in q.queued()],
            "runs": [
                {
                    "run_id": r.run_id,
                    "model": r.model,
                    "done": r.done,
                    "total": r.total,
                    "resolved": r.resolved,
                    "scored": r.scored,
                }
                for r in data.runs()
            ],
        }
    )


async def raw_summary(request: Request) -> PlainTextResponse:
    path = results_root() / request.path_params["run_id"] / "summary.md"
    if not path.exists():
        return PlainTextResponse("no summary.md; score the run first", status_code=404)
    return PlainTextResponse(path.read_text())


def build_app(host: str = "127.0.0.1") -> Starlette:
    """Serve private case data only over loopback; use an SSH tunnel for remote access."""
    if not config_io.is_loopback(host):
        raise ValueError("The dashboard requires a loopback host; use 127.0.0.1 and an SSH tunnel.")
    routes = [
        Route("/", cases_page),
        Route("/launch", launch, methods=["POST"]),
        Route("/runs", runs_page),
        Route("/runs/{run_id}", run_page),
        Route("/runs/{run_id}/panel", run_panel),
        Route("/runs/{run_id}/score", score_run, methods=["POST"]),
        Route("/runs/{run_id}/summary.md", raw_summary),
        Route("/runs/{run_id}/case/{case_id}", case_page),
        Route("/cases/{case_id}", case_across_page),
        Route("/datasets", datasets_page),
        Route("/datasets/create", dataset_create, methods=["POST"]),
        Route("/datasets/{name}/edit", dataset_edit, methods=["POST"]),
        Route("/datasets/{name}/review", dataset_review, methods=["POST"]),
        Route("/leaderboard", leaderboard_page),
        Route("/setup", setup_page),
        # Adding and editing are the same handler; the difference is whether a key was
        # opened on, which is what tells a rename to remove the name it replaced.
        Route("/setup/models", model_save, methods=["POST"]),
        Route("/setup/models/{key}", model_save, methods=["POST"]),
        Route("/setup/models/{key}/delete", model_delete, methods=["POST"]),
        Route("/setup/models-from-example", models_from_example, methods=["POST"]),
        Route("/setup/repos/detect", repo_detect, methods=["POST"]),
        Route("/setup/repos/{key}", repo_save, methods=["POST"]),
        Route("/setup/config/{name}/raw", config_raw_save, methods=["POST"]),
        Route("/compare", compare_page),
        Route("/compare/run", compare_launch, methods=["POST"]),
        Route("/jobs/{job_id}", job_page),
        Route("/jobs/{job_id}/log", job_log),
        Route("/jobs/{job_id}/cancel", job_cancel, methods=["POST"]),
        Route("/queue", queue_strip),
        Route("/api/state", api_state),
        Mount("/static", StaticFiles(directory=str(HERE / "static")), name="static"),
    ]

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        yield
        jobs.queue().shutdown()  # a running child would otherwise outlive the server

    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(DashboardBoundary)
    app.state.bind_host = host
    app.state.can_write = config_io.is_loopback(host)
    return app


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    app = build_app(host)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
