"""Check the things that otherwise fail silently, hours into a run.

Every check here exists because it cost real time: an unset repository path that only surfaced
at the first `git archive`, a missing API key that skipped a model entirely, two CLI versions in
one comparison, a provider session limit that arrived as fourteen model failures. None of them
announced itself; all of them are visible before a run starts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

Status = Literal["ok", "warn", "fail"]


# What acting on a check actually involves. The page groups by this, because "give the VM
# more memory" and "add a line to models.yaml" are not the same kind of instruction.
Kind = Literal["machine", "config", "secret", "data"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""
    kind: Kind = "machine"

    @property
    def mark(self) -> str:
        return {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}[self.status]


def _run(*argv: str, timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)
    return p.returncode, (p.stdout or p.stderr).strip()


def check_data_root() -> list[Check]:
    from eval_harness.paths import ENV_VAR, cases_dir, data_root, results_root

    out: list[Check] = []
    root = data_root()
    where = f"{root} (from {ENV_VAR})" if os.environ.get(ENV_VAR) else f"{root} (default)"
    if not root.exists():
        out.append(Check("data root", "fail", where, f"mkdir -p {root}", kind="data"))
        return out
    writable = os.access(root, os.W_OK)
    out.append(
        Check(
            "data root",
            "ok" if writable else "fail",
            where,
            "" if writable else "no write permission",
            kind="data",
        )
    )

    cases = cases_dir()
    n = len(list(cases.glob("*.json"))) if cases.is_dir() else 0
    out.append(
        Check(
            "cases",
            "ok" if n else "warn",
            f"{n} case(s) in {cases}",
            "" if n else "eval-harness collect, or eval-harness import swebench",
            kind="data",
        )
    )
    res = results_root()
    runs = (
        len([d for d in res.iterdir() if d.is_dir() and not d.name.startswith("_")])
        if res.is_dir()
        else 0
    )
    out.append(Check("results", "ok", f"{runs} run(s) in {res}", kind="data"))
    return out


def check_docker() -> list[Check]:
    if not shutil.which("docker"):
        return [Check("docker", "fail", "not on PATH", "install Docker and start it")]
    code, out = _run("docker", "info", "--format", "{{.ServerVersion}} {{.MemTotal}}")
    if code != 0:
        return [
            Check(
                "docker",
                "fail",
                out.splitlines()[0][:80] if out else "daemon unreachable",
                "start Docker",
            )
        ]
    parts = out.split()
    # With the daemon down, `docker info --format` still exits 0 on some versions (27.x on
    # macOS): the error goes to stderr and stdout is " 0" — the memory field alone. A daemon
    # that answered prints both fields.
    if len(parts) < 2:
        return [Check("docker", "fail", "daemon unreachable", "start Docker")]
    version = parts[0]
    checks = [Check("docker", "ok", f"daemon {version}")]
    try:
        gb = int(parts[1]) / 1_000_000_000
        # One case container is capped around 6 GB; two plus a prep container will not fit in 8.
        checks.append(
            Check(
                "docker memory",
                "ok" if gb >= 16 else "warn",
                f"{gb:.0f} GB",
                "" if gb >= 16 else "give the VM 16 GB or more, or run with --concurrency 1",
            )
        )
    except (IndexError, ValueError):
        pass
    return checks


def check_repos() -> list[Check]:
    from eval_harness.config import load_repos, local_path_env_var
    from eval_harness.paths import config_file

    path = config_file("repos.yaml")
    if not path.exists():
        return [
            Check(
                "repos.yaml",
                "fail",
                f"not found at {path}",
                "eval-harness init --repo <path to your repository>",
                kind="config",
            )
        ]
    try:
        repos = load_repos()
    except Exception as e:
        return [
            Check(
                "repos.yaml",
                "fail",
                f"{path}: {str(e)[:100]}",
                "fix the YAML, or regenerate with eval-harness init --force",
                kind="config",
            )
        ]
    if not repos:
        return [
            Check(
                "repos.yaml",
                "fail",
                f"{path} defines no repositories",
                "eval-harness init --repo <path to your repository>",
                kind="config",
            )
        ]

    out: list[Check] = []
    for key, repo in repos.items():
        path = repo.local_path
        # `.git` is a file, not a directory, in a git worktree; ask git instead.
        inside = path.is_dir() and _run(
            "git", "-C", str(path), "rev-parse", "--is-inside-work-tree"
        ) == (0, "true")
        if not inside:
            out.append(
                Check(
                    f"repo {key}",
                    "fail",
                    f"{path} is not a git checkout",
                    f"clone it there, or set {local_path_env_var(key)}",
                )
            )
            continue
        code, _ = _run("git", "-C", str(path), "rev-parse", "--verify", "HEAD")
        out.append(
            Check(
                f"repo {key}",
                "ok" if code == 0 else "fail",
                str(path),
                "" if code == 0 else "git rev-parse failed; is the checkout intact?",
            )
        )
        code, _ = _run("docker", "image", "inspect", repo.image)
        out.append(
            Check(
                f"image {key}",
                "ok" if code == 0 else "warn",
                repo.image,
                "" if code == 0 else f"built on first run from {repo.dockerfile}",
            )
        )
    return out


def check_tickets(online: bool = False) -> list[Check]:
    """Where each repository's tickets come from, and whether that source can be reached.

    A case is a merged pull request joined to the ticket that asked for it, so collecting needs
    the tracker as much as the repository. `linear_team` in repos.yaml is the switch, and nothing
    said so: set, tickets come from Linear and need a key in the environment; empty, they come
    from GitHub Issues through `gh`. Either way the first thing that mentioned the requirement
    used to be the crash.
    """
    from eval_harness.config import load_repos
    from eval_harness.paths import config_file

    try:
        repos = load_repos() if config_file("repos.yaml").exists() else {}
    except Exception:
        repos = {}
    # No repositories, or a file that will not load: check_repos already says so, and saying it
    # twice makes the worklist longer without making it clearer.
    if not repos:
        return []

    out: list[Check] = []
    for key, repo in repos.items():
        if repo.linear_team:
            env = repo.linear_api_key_env
            present = bool(os.environ.get(env))
            # Set or not set. Never the value, never a prefix, never a length.
            problem = None
            if present and online:
                from eval_harness.collect.linear import probe_team

                problem = probe_team(repo.linear_team, env)
            if not present:
                detail = f"Linear team {repo.linear_team}; {env} is not set"
            elif problem:
                detail = f"Linear team {repo.linear_team}; {problem}"
            else:
                detail = f"Linear team {repo.linear_team}; {env} " + (
                    "reaches it" if online else "is set — not checked against Linear"
                )
            out.append(
                Check(
                    f"tickets {key}",
                    "ok" if present and not problem else "fail",
                    detail,
                    ""
                    if present and not problem
                    else f"export {env}=… (a key from the workspace that owns "
                    f"{repo.linear_team}) in the shell that runs the harness; a personal key "
                    "is made in Linear under Settings > API > Personal API keys. Two "
                    "workspaces? set linear_api_key_env in repos.yaml",
                    kind="secret",
                )
            )
        else:
            out.append(
                Check(
                    f"tickets {key}",
                    "ok",
                    f"GitHub Issues, via gh; repos.yaml gives {key} no linear_team",
                    kind="config",
                )
            )

    # Every collect shells out to `gh` for the pull requests themselves, whichever tracker holds
    # the tickets, so this belongs to all of them rather than to the GitHub Issues ones. Presence
    # only, and it says so: `gh auth status` is a round trip to GitHub on every render of this
    # page and prints a masked token this has no business reading, while an unauthenticated gh
    # already fails with its own message — a missing one is the bare FileNotFoundError.
    exe = shutil.which("gh")
    ver = _run("gh", "--version")[1].splitlines()[0][:40] if exe else ""
    if exe and online:
        # gh answers a search on a repository it cannot see with an empty list, not an
        # error, so a login that lacks access used to look like a repository with no PRs.
        for key, repo in repos.items():
            code, msg = _run("gh", "repo", "view", repo.github, "--json", "name")
            out.append(
                Check(
                    f"github {key}",
                    "ok" if code == 0 else "fail",
                    repo.github
                    if code == 0
                    else f"gh cannot see {repo.github}: {msg.splitlines()[0][:80] if msg else ''}",
                    ""
                    if code == 0
                    else "gh auth status; gh auth switch to an account with access, or "
                    "export GH_TOKEN for this shell",
                    kind="secret",
                )
            )
    out.append(
        Check(
            "gh",
            "ok" if exe else "warn",
            f"{ver or 'on PATH'}; not checked for a login" if exe else "not on PATH",
            ""
            if exe
            else "install the GitHub CLI and run gh auth login; collecting needs it, "
            "running cases already collected does not",
            kind="machine",
        )
    )
    return out


def check_providers() -> list[Check]:
    """Which providers could run right now. Cheap probes only — no model is called."""
    from eval_harness.config import load_models
    from eval_harness.paths import config_dir, config_file

    path = config_file("models.yaml")
    if not path.exists():
        return [
            Check(
                "models.yaml",
                "fail",
                f"not found at {path}",
                f"cp config/models.example.yaml {config_dir() / 'models.yaml'}",
                kind="config",
            )
        ]
    try:
        models = load_models()
    except Exception as e:
        return [
            Check("models.yaml", "fail", f"{path}: {str(e)[:100]}", "fix the YAML", kind="config")
        ]

    providers = {m.provider for m in models.values()}
    out: list[Check] = []
    if "claude-code" in providers:
        exe = shutil.which("claude")
        ver = _run("claude", "--version")[1].splitlines()[0][:40] if exe else ""
        out.append(
            Check(
                "provider claude-code",
                "ok" if exe else "fail",
                ver or "claude not on PATH",
                "" if exe else "install Claude Code and log in",
            )
        )
    if "codex-cli" in providers:
        exe = shutil.which("codex")
        ver = _run("codex", "--version")[1].splitlines()[0][:40] if exe else ""
        out.append(
            Check(
                "provider codex-cli",
                "ok" if exe else "fail",
                ver or "codex not on PATH",
                "" if exe else "install the Codex CLI and log in",
            )
        )
    seen: set[str] = set()
    for m in models.values():
        for var in (m.api_key_env, "ANTHROPIC_API_KEY" if m.provider == "anthropic" else None):
            if not var or var in seen:
                continue
            seen.add(var)
            present = bool(os.environ.get(var))
            # The name, never the value.
            out.append(
                Check(
                    f"env {var}",
                    "ok" if present else "warn",
                    "set" if present else "not set",
                    "" if present else f"export {var}=… in the shell that runs the harness",
                    kind="secret",
                )
            )
    judge = models.get("judge")
    if judge:
        out.append(Check("judge", "ok", f"{judge.model} via {judge.provider}", kind="config"))
    return out


def run_all(online: bool = False) -> list[Check]:
    """Every check. `online` adds round trips to GitHub and Linear, which the CLI makes and the
    Setup page, which runs this on every render, does not."""
    checks: list[Check] = []
    for fn in (
        check_data_root,
        check_docker,
        check_repos,
        lambda: check_tickets(online),
        check_providers,
    ):
        try:
            checks.extend(fn())
        except Exception as e:
            name = getattr(fn, "__name__", "check")
            checks.append(Check(name, "fail", f"check itself failed: {e}"[:120]))
    return checks


def worst(checks: list[Check]) -> Status:
    if any(c.status == "fail" for c in checks):
        return "fail"
    return "warn" if any(c.status == "warn" for c in checks) else "ok"
