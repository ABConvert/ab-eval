from __future__ import annotations

import subprocess

from eval_harness.config import RepoConfig
from eval_harness.harness.sandbox import Docker, DockerError


def archive_tar(repo: RepoConfig, commit: str) -> bytes:
    """Tar of the tree at `commit`, read from the local clone without touching its worktree."""
    proc = subprocess.run(
        ["git", "-C", str(repo.local_path), "archive", "--format=tar", commit],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise DockerError(f"git archive {commit} failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def populate_source(docker: Docker, cid: str, tar: bytes) -> None:
    docker.cp_tar_in(cid, tar, "/app")


def git_init(docker: Docker, cid: str) -> str:
    res = docker.exec(
        cid,
        "git init -q && printf '.abeval*\\n' >> .git/info/exclude"
        " && git add -A && git commit -q -m base && git rev-parse HEAD",
    )
    if not res.ok:
        raise DockerError(f"git init failed: {res.stderr[-2000:]}")
    return res.stdout.strip().splitlines()[-1]


def git_commit_all(docker: Docker, cid: str, message: str) -> str:
    res = docker.exec(
        cid,
        f"git add -A && git commit -q --allow-empty -m {message!r} && git rev-parse HEAD",
    )
    if not res.ok:
        raise DockerError(f"git commit failed: {res.stderr[-2000:]}")
    return res.stdout.strip().splitlines()[-1]


def apply_patch(docker: Docker, cid: str, patch: str, message: str) -> str:
    docker.write_file(cid, "/tmp/abeval.patch", patch)
    res = docker.exec(cid, "git apply --index /tmp/abeval.patch")
    if not res.ok:
        raise DockerError(f"patch did not apply: {res.stderr[-2000:]}")
    return git_commit_all(docker, cid, message)


def diff_head(docker: Docker, cid: str) -> str:
    docker.exec(cid, "git add -A -N")  # intent-to-add so new files appear in the diff
    return docker.exec(cid, "git diff HEAD", timeout=120).stdout


def changed_paths(docker: Docker, cid: str) -> list[str]:
    out = docker.exec(cid, "git status --porcelain --untracked-files=all").stdout
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def restore_paths(docker: Docker, cid: str, paths: list[str]) -> None:
    if paths:
        docker.exec(cid, ["git", "checkout", "HEAD", "--", *paths])
