from __future__ import annotations

import contextlib
import hashlib
import subprocess
import threading

from eval_harness.config import DepDir, RepoConfig
from eval_harness.harness.sandbox import Docker, DockerError

READY = ".abeval-ready"

# Cases that share a lockfile share a dependency volume. Two of them preparing it at once
# race: each deletes and recreates the volume the other is installing into, and npm's own
# download cache collides inside it (mongodb-memory-server renames a .downloading file that
# its neighbour has already moved). One preparation at a time per volume, then — per volume
# rather than per case, because two cases can share `web/frontend` while differing on `web`.
_PREP_LOCKS: dict[str, threading.Lock] = {}
_PREP_LOCKS_GUARD = threading.Lock()


def _prep_lock(key: str) -> threading.Lock:
    with _PREP_LOCKS_GUARD:
        return _PREP_LOCKS.setdefault(key, threading.Lock())


def lock_hash(repo: RepoConfig, dep: DepDir, commit: str) -> str:
    """sha256 over the dep's lockfiles as they exist at `commit` (missing file = empty)."""
    h = hashlib.sha256()
    for lock in dep.lock:
        proc = subprocess.run(
            ["git", "-C", str(repo.local_path), "show", f"{commit}:{lock}"], capture_output=True
        )
        h.update(lock.encode() + b"\0" + (proc.stdout if proc.returncode == 0 else b"") + b"\0")
    return h.hexdigest()[:12]


def volume_name(repo_key: str, dep_dir: str, digest: str) -> str:
    return f"abeval-deps-{repo_key}-{dep_dir.replace('/', '_')}-{digest}"


def _ready(docker: Docker, image: str, volume: str) -> bool:
    if not docker.volume_exists(volume):
        return False
    probe = docker._run(
        "run", "--rm", "-v", f"{volume}:/probe", image, "test", "-f", f"/probe/{READY}"
    )
    return probe.returncode == 0


def ensure_dep_volumes(
    docker: Docker, repo: RepoConfig, commit: str, *, source_tar: bytes
) -> dict[str, str]:
    """Return {dep_dir: volume}. Missing volumes are filled by a network-enabled prep step."""
    plan = {
        dep.dir: volume_name(repo.key, dep.dir, lock_hash(repo, dep, commit))
        for dep in repo.dep_dirs
    }
    if not [dep for dep in repo.dep_dirs if not _ready(docker, repo.image, plan[dep.dir])]:
        return plan
    # Sorted, so two cases wanting the same pair of volumes cannot deadlock on each other.
    with contextlib.ExitStack() as stack:
        for volume in sorted(set(plan.values())):
            stack.enter_context(_prep_lock(volume))
        return _prepare(docker, repo, commit, plan, source_tar=source_tar)


def _prepare(
    docker: Docker,
    repo: RepoConfig,
    commit: str,
    plan: dict[str, str],
    *,
    source_tar: bytes,
) -> dict[str, str]:
    # Re-check under the lock: whoever held it may have filled these volumes already.
    missing = [dep for dep in repo.dep_dirs if not _ready(docker, repo.image, plan[dep.dir])]
    if not missing:
        return plan
    for dep in missing:
        docker.volume_rm(plan[dep.dir])
        docker.volume_create(plan[dep.dir])
    mounts = {plan[dep.dir]: f"/app/{dep.dir}/node_modules" for dep in missing}
    name = f"abeval-prep-{commit[:10]}"
    docker.rm(name)  # a killed job can leave a stale container holding this name
    cid = docker.create_container(
        image=repo.image,
        name=name,
        mounts=mounts,
        limits=repo.limits,
        network=True,
    )
    try:
        docker.start(cid)
        docker.cp_tar_in(cid, source_tar, "/app")
        for dep in missing:
            env = {"CI": "1", **dep.env}
            steps = [("install", dep.install)]
            if dep.post_install:
                steps.append(("post_install", dep.post_install))
            for label, cmd in steps:
                res = docker.exec(cid, cmd, cwd=f"/app/{dep.dir}", timeout=1800, env=env)
                if not res.ok:
                    docker.volume_rm(plan[dep.dir])
                    raise DockerError(f"{label} failed in {dep.dir}: {res.stderr[-3000:]}")
            docker.exec(cid, f"touch node_modules/{READY}", cwd=f"/app/{dep.dir}")
    finally:
        docker.rm(cid)
    return plan
