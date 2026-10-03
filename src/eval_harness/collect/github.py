from __future__ import annotations

import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

CHUNK_DAYS = 7  # 14-day windows with `files` made GitHub's search backend return 502s
PAGE_LIMIT = 600
RETRIES = 5
FIELDS = (
    "number,title,body,headRefName,author,mergeCommit,mergedAt,createdAt,files,reviews,comments"
)

T = TypeVar("T", bound=BaseModel)


class PRFile(BaseModel):
    path: str
    additions: int
    deletions: int


class PullRequest(BaseModel):
    number: int
    title: str
    body: str
    head_ref: str
    author: str
    merged_at: str
    created_at: str
    merge_commit: str
    files: list[PRFile]
    review_count: int
    comment_count: int


def windows(since: str, until: str, chunk_days: int = CHUNK_DAYS) -> list[tuple[str, str]]:
    """Inclusive [since, until] split into chunks of at most `chunk_days` days."""
    start = date.fromisoformat(since)
    end = date.fromisoformat(until)
    out: list[tuple[str, str]] = []
    while start <= end:
        stop = min(start + timedelta(days=chunk_days - 1), end)
        out.append((start.isoformat(), stop.isoformat()))
        start = stop + timedelta(days=1)
    return out


def save_cache(cache_dir: Path, name: str, items: list[T]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.json"
    path.write_text(json.dumps([i.model_dump() for i in items], indent=1) + "\n")
    return path


def load_cache(cache_dir: Path, name: str, model: type[T]) -> list[T] | None:
    path = cache_dir / f"{name}.json"
    if not path.exists():
        return None
    return [model.model_validate(x) for x in json.loads(path.read_text())]


def _from_gh(raw: dict[str, Any]) -> PullRequest:
    return PullRequest(
        number=int(raw["number"]),
        title=raw.get("title") or "",
        body=raw.get("body") or "",
        head_ref=raw.get("headRefName") or "",
        author=(raw.get("author") or {}).get("login") or "",
        merged_at=raw.get("mergedAt") or "",
        created_at=raw.get("createdAt") or "",
        merge_commit=((raw.get("mergeCommit") or {}).get("oid")) or "",
        files=[
            PRFile(path=f["path"], additions=int(f["additions"]), deletions=int(f["deletions"]))
            for f in raw.get("files") or []
        ],
        review_count=len(raw.get("reviews") or []),
        comment_count=len(raw.get("comments") or []),
    )


def _gh_window(github: str, start: str, stop: str) -> list[dict[str, Any]]:
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        github,
        "--state",
        "merged",
        "--search",
        f"merged:{start}..{stop}",
        "--limit",
        str(PAGE_LIMIT),
        "--json",
        FIELDS,
    ]
    last_err = ""
    for attempt in range(RETRIES):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                data: list[dict[str, Any]] = json.loads(proc.stdout)
                return data
            except json.JSONDecodeError as e:
                last_err = f"bad JSON: {e}"
        else:
            last_err = proc.stderr[-500:]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gh pr list failed for {start}..{stop}: {last_err}")


def fetch_merged_prs(
    github: str, since: str, until: str, *, cache_dir: Path, refresh: bool = False
) -> list[PullRequest]:
    name = f"prs-{since}-{until}"
    if not refresh:
        cached = load_cache(cache_dir, name, PullRequest)
        if cached is not None:
            return cached
    seen: dict[int, PullRequest] = {}
    for start, stop in windows(since, until):
        rows = _gh_window(github, start, stop)
        if len(rows) >= PAGE_LIMIT:
            raise RuntimeError(
                f"window {start}..{stop} returned {PAGE_LIMIT} PRs; shrink CHUNK_DAYS"
            )
        for raw in rows:
            pr = _from_gh(raw)
            seen[pr.number] = pr
    prs = sorted(seen.values(), key=lambda p: p.number)
    # An empty answer is not cached. `gh` returns [] rather than an error when the signed-in
    # account cannot see a private repository, and a cached [] then reports "0 merged PRs"
    # on every later run, long after the login is fixed.
    if prs:
        save_cache(cache_dir, name, prs)
    return prs
