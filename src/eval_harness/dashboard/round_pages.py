"""Round workflows over the existing queue and analytics."""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.parse import quote

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from eval_harness import rounds
from eval_harness.collect import datasets
from eval_harness.config import ModelConfig, load_models
from eval_harness.dashboard import data, jobs
from eval_harness.paths import results_root
from eval_harness.report.leaderboard import build


def _round(request: Request) -> rounds.Round:
    try:
        return rounds.load(request.path_params["round_id"])
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, "Round not found") from e


def eligible(round_: rounds.Round) -> list[data.RunView]:
    """Latest complete scored attempt per configuration; controls are never ranked."""
    selected = []
    for entry in round_.entries:
        if entry.model.provider == "human-patch" or entry.model.key == "human-patch":
            continue
        for rid in reversed(entry.run_ids):
            view = data.load_run(rid)
            if not view or not view.summary or not view.finished_at:
                continue
            summary = view.summary
            meta = json.loads((results_root() / rid / "run.json").read_text())
            if (
                meta.get("round_id") == round_.id
                and meta.get("cases") == [c.case_id for c in round_.cases]
                and summary.run_id == rid
                and summary.model == entry.model.key
                and len(summary.cases) == len(round_.cases)
                and {c.case_id for c in summary.cases} == {c.case_id for c in round_.cases}
                and summary.caps == round_.caps.model_dump()
                and summary.model_config_snapshot == entry.model.model_dump(mode="json")
                and summary.harness_git_sha == round_.harness_sha
                and summary.judge_config == round_.judge.model_dump(mode="json")
                and all(c.status == "completed" for c in summary.cases)
                and all(c.code_quality is not None for c in summary.cases if c.resolved)
            ):
                selected.append(view)
                break
    return selected


def queue_attempt(request: Request, round_: rounds.Round, entry: rounds.Entry, rid: str) -> None:
    jobs.queue().submit(
        jobs.Job(
            id=uuid.uuid4().hex[:12],
            kind="run",
            label=f"{entry.model.key} · {round_.name}",
            argv=jobs.harness_argv(
                "run", "--model", entry.model.key, "--round-id", round_.id, "--run-id", rid
            ),
            run_id=rid,
            cases=[c.case_id for c in round_.cases],
        )
    )


def _context(request: Request, **kw: Any) -> dict[str, Any]:
    from eval_harness.dashboard.app import _ctx

    return _ctx(request, **kw)


async def index(request: Request) -> HTMLResponse:
    from eval_harness.dashboard.app import templates

    return templates.TemplateResponse(
        request,
        "rounds.html",
        _context(request, rounds=rounds.all_rounds(), page="rounds"),
    )


async def new(request: Request) -> HTMLResponse:
    from eval_harness.dashboard.app import templates

    try:
        models = load_models()
        models["human-patch"] = ModelConfig(
            key="human-patch", provider="human-patch", model="human-patch"
        )
    except (OSError, ValueError):
        models = {}
    source = request.query_params.get("source", "")
    original = None
    if source:
        try:
            original = rounds.load(source)
        except (OSError, ValueError) as e:
            raise HTTPException(404, "Round not found") from e
    return templates.TemplateResponse(
        request,
        "round_new.html",
        _context(
            request,
            datasets=datasets.load_all(),
            models=models,
            original=original,
            error=request.query_params.get("error", ""),
            page="rounds",
        ),
    )


async def create(request: Request) -> RedirectResponse:
    form = await request.form()
    try:
        keys = [str(k) for k in form.getlist("models")]
        models = load_models()
        if (
            not keys
            or len(set(keys)) != len(keys)
            or any(
                (k not in models and k != "human-patch") or k in ("judge", "classifier")
                for k in keys
            )
        ):
            raise ValueError("Choose at least one configured candidate model")
        round_ = rounds.create(
            str(form.get("name", "")),
            str(form.get("dataset", "")),
            source=str(form.get("source", "")) or None,
        )
        for entry, rid in rounds.append_models(round_.id, keys):
            queue_attempt(request, round_, entry, rid)
    except (OSError, ValueError, KeyError) as e:
        return RedirectResponse(f"/rounds/new?error={quote(str(e))}", status_code=303)
    return RedirectResponse(f"/rounds/{round_.id}", status_code=303)


async def detail(request: Request) -> HTMLResponse:
    from eval_harness.dashboard.app import templates

    round_ = _round(request)
    try:
        models = load_models()
        models["human-patch"] = ModelConfig(
            key="human-patch", provider="human-patch", model="human-patch"
        )
    except (OSError, ValueError):
        models = {}
    available = {
        k: m
        for k, m in models.items()
        if k not in ("judge", "classifier") and all(m != e.model for e in round_.entries)
    }
    attempts: list[dict[str, Any]] = []
    history = {j.run_id: j for j in jobs.queue().recent(100000)}
    for entry in round_.entries:
        for rid in reversed(entry.run_ids):
            job = history.get(rid)
            view = data.load_run(rid)
            attempts.append(
                {
                    "entry": entry,
                    "run_id": rid,
                    "view": view,
                    "job": job,
                    "latest": rid == entry.run_ids[-1],
                }
            )
    active = any(a["job"] and a["job"].active for a in attempts)
    ranked = eligible(round_)
    complete = len(ranked) == len([e for e in round_.entries if e.model.key != "human-patch"])
    complete = complete and all(
        any(a["entry"].id == e.id and a["view"] and a["view"].scored for a in attempts)
        for e in round_.entries
        if e.model.key == "human-patch"
    )
    return templates.TemplateResponse(
        request,
        "round_detail.html",
        _context(
            request,
            round=round_,
            attempts=attempts,
            models=available,
            eligible=ranked,
            active=active,
            complete=complete,
            error=request.query_params.get("error", ""),
            page="rounds",
        ),
    )


async def add(request: Request) -> RedirectResponse:
    round_ = _round(request)
    form = await request.form()
    try:
        for entry, rid in rounds.append_models(round_.id, [str(k) for k in form.getlist("models")]):
            queue_attempt(request, round_, entry, rid)
    except (OSError, ValueError) as e:
        return RedirectResponse(f"/rounds/{round_.id}?error={quote(str(e))}", status_code=303)
    return RedirectResponse(f"/rounds/{round_.id}", status_code=303)


async def retry(request: Request) -> RedirectResponse:
    round_ = _round(request)
    entry = next((e for e in round_.entries if e.id == request.path_params["entry_id"]), None)
    if entry is None:
        raise HTTPException(404, "Model not found")
    if any(jobs.queue().active_for_run(rid) for rid in entry.run_ids):
        raise HTTPException(409, "Wait for this model's current attempt to finish")
    try:
        entry, rid = rounds.retry(round_.id, entry.id)
        queue_attempt(request, round_, entry, rid)
    except ValueError as e:
        return RedirectResponse(f"/rounds/{round_.id}?error={quote(str(e))}", status_code=303)
    return RedirectResponse(f"/rounds/{round_.id}", status_code=303)


async def leaderboard(request: Request) -> HTMLResponse:
    from eval_harness.dashboard.app import _series_order, templates

    round_ = _round(request)
    scored = eligible(round_)
    return templates.TemplateResponse(
        request,
        "leaderboard.html",
        _context(
            request,
            round=round_,
            round_choices=rounds.all_rounds(),
            board=build([r.summary for r in scored if r.summary], weights=round_.weights),
            datasets=[],
            wanted=round_.name,
            mixed_warning=False,
            run_datasets={r.run_id: round_.name for r in scored},
            headline={
                "models": len(scored),
                "cases": len(round_.cases),
                "attempts": sum(r.total for r in scored),
                "description": "Frozen benchmark",
            },
            series_order=_series_order(),
            page="rounds",
        ),
    )
