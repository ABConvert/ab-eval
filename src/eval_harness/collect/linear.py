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


def api_key(env: str = LINEAR_API_KEY_ENV) -> str:
    """The key Linear is called with, or a sentence saying how to get one.

    Only a repository with a `linear_team` in repos.yaml comes through here, and nothing earlier
    in a collect touches Linear — so an unset variable used to arrive as a bare KeyError, minutes
    in, naming nothing. The message is the whole point of this function; the value it returns is
    never logged, cached or written to a config file.
    """
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(
            f"{env} is not set, and this repository reads its tickets from Linear "
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


def _post(
    query: str, variables: dict[str, Any], *, key_env: str = LINEAR_API_KEY_ENV
) -> dict[str, Any]:
    resp = httpx.post(
        LINEAR_URL,
        headers={"Authorization": api_key(key_env), "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=60,
    )
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"Linear error: {payload['errors'][:1]}")
    return payload


def fetch_issues(
    team: str,
    numbers: list[int],
    *,
    cache_dir: Path,
    refresh: bool = False,
    key_env: str = LINEAR_API_KEY_ENV,
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
        payload = _post(
            QUERY, {"team": team, "numbers": [float(n) for n in chunk]}, key_env=key_env
        )
        issues.extend(_from_linear(n) for n in payload["data"]["issues"]["nodes"])
    save_cache(cache_dir, name, issues)
    return {i.key: i for i in issues}


TEAM_QUERY = "query($key: String!) { teams(filter: {key: {eq: $key}}) { nodes { id key } } }"


def probe_team(team: str, key_env: str = LINEAR_API_KEY_ENV) -> str | None:
    """None when the key reaches `team`; otherwise a sentence saying what is wrong.

    A key from another workspace is valid, so only asking for the team tells the two apart,
    and that difference used to surface at collect time as "ticket not found".
    """
    try:
        payload = _post(TEAM_QUERY, {"key": team}, key_env=key_env)
    except httpx.HTTPStatusError as e:
        return f"Linear rejected {key_env} (HTTP {e.response.status_code})"
    except (httpx.HTTPError, RuntimeError) as e:
        return f"could not reach Linear: {str(e)[:100]}"
    if not payload.get("data", {}).get("teams", {}).get("nodes"):
        return (
            f"{key_env} works, but its workspace has no team {team}: "
            "is it a key for a different Linear workspace?"
        )
    return None
