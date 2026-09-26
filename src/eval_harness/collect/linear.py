from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from eval_harness.collect.github import load_cache, save_cache
from eval_harness.config import LINEAR_API_KEY_ENV

LINEAR_URL = "https://api.linear.app/graphql"
BATCH = 40
QUERY = """
query($team: String!, $numbers: [Float!]) {
  issues(first: 50, filter: { team: { key: { eq: $team } }, number: { in: $numbers } }) {
    nodes {
      number identifier title description createdAt startedAt completedAt
      labels { nodes { name } }
      comments { nodes { body } }
      children { nodes { identifier } }
    }
  }
}
"""


class Issue(BaseModel):
    key: str
    number: int
    title: str
    description: str
    labels: list[str]
    created_at: str
    started_at: str | None
    completed_at: str | None
    comments: list[str]
    children: list[str]


def api_key() -> str:
    """The key Linear is called with, or a sentence saying how to get one.

    Only a repository with a `linear_team` in repos.yaml comes through here, and nothing earlier
    in a collect touches Linear — so an unset variable used to arrive as a bare KeyError, minutes
    in, naming nothing. The message is the whole point of this function; the value it returns is
    never logged, cached or written to a config file.
    """
    key = os.environ.get(LINEAR_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{LINEAR_API_KEY_ENV} is not set, and this repository reads its tickets from Linear "
            f"(repos.yaml gives it a linear_team). Export it in the shell that runs the harness "
            f"and try again — a personal key is made in Linear under Settings > API > Personal "
            f"API keys. The Setup page reports whether it is set."
        )
    return key


def _from_linear(node: dict[str, Any]) -> Issue:
    return Issue(
        key=node["identifier"],
        number=int(node["number"]),
        title=node.get("title") or "",
        description=node.get("description") or "",
        labels=[x["name"] for x in (node.get("labels") or {}).get("nodes", [])],
        created_at=node["createdAt"],
        started_at=node.get("startedAt"),
        completed_at=node.get("completedAt"),
        comments=[c["body"] for c in (node.get("comments") or {}).get("nodes", [])],
        children=[c["identifier"] for c in (node.get("children") or {}).get("nodes", [])],
    )


def _post(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    resp = httpx.post(
        LINEAR_URL,
        headers={"Authorization": api_key(), "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=60,
    )
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"Linear error: {payload['errors'][:1]}")
    return payload


def fetch_issues(
    team: str, numbers: list[int], *, cache_dir: Path, refresh: bool = False
) -> dict[str, Issue]:
    wanted = sorted(set(numbers))
    digest = hashlib.sha256(",".join(map(str, wanted)).encode()).hexdigest()[:12]
    name = f"linear-{team}-{digest}"
    if not refresh:
        cached = load_cache(cache_dir, name, Issue)
        if cached is not None:
            return {i.key: i for i in cached}
    issues: list[Issue] = []
    for i in range(0, len(wanted), BATCH):
        chunk = wanted[i : i + BATCH]
        payload = _post(QUERY, {"team": team, "numbers": [float(n) for n in chunk]})
        issues.extend(_from_linear(n) for n in payload["data"]["issues"]["nodes"])
    save_cache(cache_dir, name, issues)
    return {i.key: i for i in issues}
