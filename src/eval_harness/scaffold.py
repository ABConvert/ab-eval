"""Read a repository and propose a first `repos.yaml` for it.

Nobody writes this file correctly from a README. It wants dependency roots with their
lockfiles, a runner per test suite with its config and globs, and a base image with the right
toolchain. All of that is visible in the repository, so read it and write a draft the owner
corrects — every guess carries a CHECK marker so it is obvious what to look at.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Directories that never hold a project's own source.
SKIP = {
    "node_modules",
    ".git",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".nuxt",
    "vendor",
    "target",
    "coverage",
    ".tox",
    ".mypy_cache",
}
MAX_DEPTH = 3


@dataclass
class DetectedDep:
    """One directory whose dependencies install together."""

    dir: str
    install: str
    lock: list[str]


@dataclass
class DetectedRunner:
    """One test suite: how to run it and which files belong to it."""

    name: str
    kind: str
    cwd: str
    config: str | None
    match: list[str]
    full: str


@dataclass
class Detection:
    key: str
    github: str | None
    local_path: str
    image: str
    dockerfile: str
    dep_dirs: list[DetectedDep] = field(default_factory=list)
    runners: list[DetectedRunner] = field(default_factory=list)
    ticket_prefix: str | None = None
    notes: list[str] = field(default_factory=list)


def _walk(root: Path, name: str) -> list[Path]:
    """Every `name` under root, pruning vendored trees and stopping at MAX_DEPTH.

    Pruned during the walk, not filtered after it: a repository with node_modules holds
    hundreds of thousands of files, and rglob would visit every one of them.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        depth = len(here.relative_to(root).parts)
        dirnames[:] = [] if depth >= MAX_DEPTH else [d for d in dirnames if d not in SKIP]
        if name in filenames:
            found.append(here / name)
    return sorted(found)


def detect_remote(repo: Path) -> str | None:
    """`owner/name` from the origin remote, for either URL form."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    url = proc.stdout.strip()
    m = re.search(r"[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
    return f"{m.group(1)}/{m.group(2)}" if m else None


NODE_LOCKS = {
    "package-lock.json": "npm ci",
    "pnpm-lock.yaml": "pnpm install --frozen-lockfile",
    "yarn.lock": "yarn install --frozen-lockfile",
    "bun.lockb": "bun install --frozen-lockfile",
}
PY_LOCKS = {"uv.lock": "uv sync --frozen", "poetry.lock": "poetry install --no-root"}


def detect_dep_dirs(repo: Path) -> list[DetectedDep]:
    """Every directory holding a manifest with a lockfile beside it."""
    out: list[DetectedDep] = []
    for manifest in _walk(repo, "package.json"):
        d = manifest.parent
        rel = d.relative_to(repo).as_posix() or "."
        for lockname, install in NODE_LOCKS.items():
            if (d / lockname).exists():
                locks = (
                    [f"{rel}/package.json", f"{rel}/{lockname}"]
                    if rel != "."
                    else ["package.json", lockname]
                )
                out.append(DetectedDep(dir=rel, install=install, lock=locks))
                break
    for manifest in _walk(repo, "pyproject.toml"):
        d = manifest.parent
        rel = d.relative_to(repo).as_posix() or "."
        for lockname, install in PY_LOCKS.items():
            if (d / lockname).exists():
                prefix = "" if rel == "." else f"{rel}/"
                out.append(
                    DetectedDep(
                        dir=rel,
                        install=install,
                        lock=[f"{prefix}pyproject.toml", f"{prefix}{lockname}"],
                    )
                )
                break
    return out


VITEST_CONFIGS = ("vitest.config.ts", "vitest.config.js", "vitest.config.mjs")
JEST_CONFIGS = ("jest.config.js", "jest.config.ts", "jest.config.mjs")


def detect_jest_dirs(repo: Path) -> list[str]:
    """Jest suites. There is no jest parser yet, so these are reported, never emitted."""
    found = []
    for cfg_name in JEST_CONFIGS:
        for cfg in _walk(repo, cfg_name):
            found.append(cfg.parent.relative_to(repo).as_posix() or ".")
    return sorted(set(found))


def detect_runners(repo: Path) -> list[DetectedRunner]:
    """One runner per test suite, with globs drawn from where the tests actually are."""
    out: list[DetectedRunner] = []
    for cfg_name in VITEST_CONFIGS:
        for cfg in _walk(repo, cfg_name):
            d = cfg.parent
            rel = d.relative_to(repo).as_posix() or "."
            globs = _test_globs(repo, d)
            if not globs:
                continue
            out.append(
                DetectedRunner(
                    name=(rel.replace("/", "-") if rel != "." else "unit"),
                    kind="vitest",
                    cwd=rel,
                    config=cfg_name,
                    match=globs,
                    full=f"npx vitest run --config {cfg_name}",
                )
            )
    for manifest in _walk(repo, "pyproject.toml"):
        try:
            data = tomllib.loads(manifest.read_text())
        except Exception:
            continue
        if "pytest" not in json.dumps(data.get("tool", {})):
            continue
        rel = manifest.parent.relative_to(repo).as_posix() or "."
        prefix = "" if rel == "." else f"{rel}/"
        out.append(
            DetectedRunner(
                name=(rel.replace("/", "-") if rel != "." else "pytest"),
                kind="pytest",
                cwd=rel,
                config=None,
                match=[f"{prefix}**/test_*.py", f"{prefix}**/*_test.py"],
                full="python -m pytest",
            )
        )
    return out


def _test_globs(repo: Path, under: Path, *, budget: int = 40_000) -> list[str]:
    """Globs covering the test files actually present beneath `under`.

    Prunes vendored directories as it goes, and stops after `budget` files: the answer is a
    small set of suffixes, and a monorepo should not cost a minute to guess them.
    """
    suffixes: set[str] = set()
    seen = 0
    for _dirpath, dirnames, filenames in os.walk(under):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for fname in filenames:
            seen += 1
            if re.search(r"\.(test|spec)\.[cm]?[jt]sx?$", fname):
                suffixes.add(fname.rsplit(".", 2)[-1])
        if seen > budget:
            break
    rel = under.relative_to(repo).as_posix()
    prefix = "" if rel == "." else f"{rel}/"
    return [f"{prefix}**/*.test.{s}" for s in sorted(suffixes)]


def detect_base_image(repo: Path) -> tuple[str, str, list[str]]:
    """(image tag, dockerfile, notes) from whatever pins a toolchain version."""
    notes: list[str] = []
    nvmrc = repo / ".nvmrc"
    if nvmrc.exists():
        major = re.sub(r"[^\d.]", "", nvmrc.read_text()).split(".")[0]
        if major:
            return f"abeval/{repo.name}:node{major}", "docker/node20.Dockerfile", notes
    pkg = repo / "package.json"
    if pkg.exists():
        try:
            engines = json.loads(pkg.read_text()).get("engines", {})
            m = re.search(r"(\d+)", str(engines.get("node", "")))
            if m:
                return f"abeval/{repo.name}:node{m.group(1)}", "docker/node20.Dockerfile", notes
        except Exception:
            pass
        notes.append("No node version pinned; assuming the node20 base image.")
        return f"abeval/{repo.name}:node20", "docker/node20.Dockerfile", notes
    if (repo / "pyproject.toml").exists():
        return f"abeval/{repo.name}:python312", "docker/python312.Dockerfile", notes
    notes.append("Could not tell what toolchain this repository needs; pick a base image.")
    return f"abeval/{repo.name}:base", "docker/node20.Dockerfile", notes


def detect_ticket_prefix(repo: Path, sample: int = 200) -> str | None:
    """The issue-key prefix branch names use, when one dominates."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%s", f"-{sample}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    hits = Counter(m.group(1) for m in re.finditer(r"\b([A-Z]{2,10})-\d+\b", proc.stdout))
    if not hits:
        return None
    top, count = hits.most_common(1)[0]
    return top if count >= 5 else None


def detect(repo: Path, key: str = "") -> Detection:
    repo = repo.expanduser().resolve()
    image, dockerfile, notes = detect_base_image(repo)
    det = Detection(
        key=key or repo.name,
        github=detect_remote(repo),
        local_path=str(repo),
        image=image,
        dockerfile=dockerfile,
        dep_dirs=detect_dep_dirs(repo),
        runners=detect_runners(repo),
        ticket_prefix=detect_ticket_prefix(repo),
        notes=notes,
    )
    if not det.dep_dirs:
        det.notes.append(
            "No manifest with a lockfile found; dep_dirs is empty and must be written by hand."
        )
    if not det.runners:
        det.notes.append(
            "No test runner config found; runners is empty and must be written by hand."
        )
    jest = detect_jest_dirs(repo)
    if jest:
        det.notes.append(
            "Jest suites found in " + ", ".join(jest) + " — there is no jest output parser yet, "
            "so they are left out. Adding one is a contained change; see CONTRIBUTING.md."
        )
    if len(det.dep_dirs) > 4:
        det.notes.append(
            f"{len(det.dep_dirs)} dependency roots found. Installing all of them is slow; keep "
            "only the ones the suites you benchmark actually need."
        )
    if det.github is None:
        det.notes.append("No origin remote; set `github:` by hand.")
    return det


def render(det: Detection) -> str:
    """A repos.yaml entry with every guess marked, so the owner knows where to look."""
    lines: list[str] = [
        f"# Written by `eval-harness init` from {det.local_path}.",
        "# Every CHECK below is a guess. Correct them, then run `eval-harness doctor`.",
        "",
        f"{det.key}:",
        f"  github: {det.github or 'OWNER/NAME   # CHECK: no origin remote was found'}",
        f"  local_path: {det.local_path}",
        f"  image: {det.image}",
        f"  dockerfile: {det.dockerfile}",
    ]
    if det.ticket_prefix:
        lines += [
            "  # Branch names here carry this issue-key prefix, so tickets come from that tracker.",
            "  # Delete this to read GitHub Issues instead.",
            f"  linear_team: {det.ticket_prefix}",
        ]
    else:
        lines += ["  # No issue-key prefix in recent commits, so tickets come from GitHub Issues."]

    lines += [
        "",
        "  # CHECK: keep only the roots whose dependencies the benchmarked suites need.",
        "  dep_dirs:",
    ]
    if det.dep_dirs:
        for d in det.dep_dirs:
            lines += [
                f"    - dir: {d.dir}",
                f'      install: "{d.install}"',
                f"      lock: [{', '.join(d.lock)}]",
            ]
    else:
        lines += ["    []   # CHECK: no manifest with a lockfile was found"]

    lines += [
        "",
        "  # CHECK: run each `full` command yourself once; it must pass on a clean checkout.",
        "  #        Order matters — a narrow runner must come before a broad one.",
        "  runners:",
    ]
    if det.runners:
        for r in det.runners:
            lines += [f"    {r.name}:", f"      kind: {r.kind}", f"      cwd: {r.cwd}"]
            if r.config:
                lines.append(f"      config: {r.config}")
            lines += [
                f"      match: [{', '.join(repr(m) for m in r.match)}]",
                f'      full: "{r.full}"',
            ]
    else:
        lines += ["    {}   # CHECK: no runner config was found"]

    lines += [
        "",
        "  test_file_globs: "
        '["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "**/test/**", "**/tests/**"]',
        "  non_code_globs: "
        '["**/*.md", "**/*.mdx", "**/*.json", "**/*.yaml", "**/*.yml", "**/*.png", "**/*.svg"]',
        "  limits: {cpus: 4, memory: 6g, pids: 4096}",
    ]
    if det.notes:
        lines += ["", "# Worth knowing:"]
        lines += [f"#   - {n}" for n in det.notes]
    return "\n".join(lines) + "\n"
