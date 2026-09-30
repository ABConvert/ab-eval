from __future__ import annotations

import contextlib
import hashlib
import posixpath
import subprocess
import sys
import threading
import time

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


def cache_volume_name(repo_key: str) -> str:
    return f"abeval-cache-{repo_key}"


def mount_path(dep: DepDir) -> str:
    """Where a dep's volume is mounted: `/app/<dir>/<target>`, normalised for `dir: .`."""
    return posixpath.normpath(posixpath.join("/app", dep.dir, dep.target))


def mounts_for(repo: RepoConfig, plan: dict[str, str]) -> dict[str, str]:
    """{volume: mount path} for a plan returned by `ensure_dep_volumes`."""
    by_dir = {d.dir: d for d in repo.dep_dirs}
    return {volume: mount_path(by_dir[d]) for d, volume in plan.items()}


# The package managers' download caches live on one volume per repository, so a prep that
# timed out, or a new lockfile that shares most of its packages with the last one, does not
# fetch everything again. Only the prep container mounts it; case containers never see it.
CACHE_MOUNT = "/root/.cache"
CACHE_ENV = {
    "UV_CACHE_DIR": f"{CACHE_MOUNT}/uv",
    "PIP_CACHE_DIR": f"{CACHE_MOUNT}/pip",
    "npm_config_cache": f"{CACHE_MOUNT}/npm",
    # The cache and the dependency volume are different filesystems, so uv cannot hardlink.
    "UV_LINK_MODE": "copy",
}


def _say(msg: str) -> None:
    # stderr, so a command that prints results on stdout keeps them clean. A cold install can
    # take many minutes, and silence for that long reads as a hang.
    print(msg, file=sys.stderr, flush=True)


def _ready(docker: Docker, image: str, volume: str) -> bool:
    if not docker.volume_exists(volume):
        return False
    probe = docker._run(
        "run", "--rm", "-v", f"{volume}:/probe", image, "test", "-f", f"/probe/{READY}"
    )
    return probe.returncode == 0


def ensure_dep_volumes(
    docker: Docker,
    repo: RepoConfig,
    commit: str,
    *,
    source_tar: bytes,
    dep_dirs: list[DepDir] | None = None,
) -> dict[str, str]:
    """Return {dep_dir: volume}. Missing volumes are filled by a network-enabled prep step.

    `dep_dirs` narrows the set to what the case's runners need (`RepoConfig.deps_for`);
    None prepares every dep_dir, as before runners could say which they use.
    """
    deps = list(repo.dep_dirs) if dep_dirs is None else dep_dirs
    plan = {dep.dir: volume_name(repo.key, dep.dir, lock_hash(repo, dep, commit)) for dep in deps}
    if not [dep for dep in deps if not _ready(docker, repo.image, plan[dep.dir])]:
        return plan
    # Sorted, so two cases wanting the same pair of volumes cannot deadlock on each other.
    with contextlib.ExitStack() as stack:
        for volume in sorted(set(plan.values())):
            stack.enter_context(_prep_lock(volume))
        return _prepare(docker, repo, commit, plan, deps, source_tar=source_tar)


def _prepare(
    docker: Docker,
    repo: RepoConfig,
    commit: str,
    plan: dict[str, str],
    deps: list[DepDir],
    *,
    source_tar: bytes,
) -> dict[str, str]:
    # Re-check under the lock: whoever held it may have filled these volumes already.
    missing = [dep for dep in deps if not _ready(docker, repo.image, plan[dep.dir])]
    if not missing:
        return plan
    for dep in missing:
        docker.volume_rm(plan[dep.dir])
        docker.volume_create(plan[dep.dir])
    mounts = {plan[dep.dir]: mount_path(dep) for dep in missing}
    cache = cache_volume_name(repo.key)
    if not docker.volume_exists(cache):
        docker.volume_create(cache)
    mounts[cache] = CACHE_MOUNT
    name = f"abeval-prep-{commit[:10]}"
    docker.rm(name)  # a killed job can leave a stale container holding this name
    cid = docker.create_container(
        image=repo.image,
        name=name,
        mounts=mounts,
        limits=repo.limits,
        network=True,
    )
    done: set[str] = set()
    drop_cache = False
    try:
        docker.start(cid)
        docker.cp_tar_in(cid, source_tar, "/app")
        for dep in missing:
            env = {"CI": "1", **CACHE_ENV, **dep.env}
            steps = [("install", dep.install)]
            if dep.post_install:
                steps.append(("post_install", dep.post_install))
            for label, cmd in steps:
                _say(f"  {label} {dep.dir}: {cmd} (timeout {dep.install_timeout}s)")
                t0 = time.monotonic()
                res = docker.exec(
                    cid, cmd, cwd=f"/app/{dep.dir}", timeout=dep.install_timeout, env=env
                )
                took = time.monotonic() - t0
                if not res.ok:
                    # A timeout keeps the cache so the retry resumes. Anything else may be
                    # the cache itself — a download cut off when Docker died mid-write is a
                    # corrupt wheel every later attempt would trip on — so start clean.
                    drop_cache = not res.timed_out
                    hint = (
                        f" — raise install_timeout for {dep.dir} in repos.yaml; the download"
                        " cache is kept, so a retry resumes"
                        if res.timed_out
                        else " — the download cache was cleared, in case it was the cause"
                    )
                    raise DockerError(
                        f"{label} failed in {dep.dir} after {took:.0f}s{hint}: {res.stderr[-3000:]}"
                    )
                _say(f"  {label} {dep.dir}: done in {took:.0f}s")
            docker.exec(cid, f"touch {READY}", cwd=mount_path(dep))
            done.add(dep.dir)
    finally:
        docker.rm(cid)
        # After the container, not before: a volume still mounted cannot be removed, and
        # `volume rm` failing quietly is how half-filled volumes used to outlive a failure.
        for dep in missing:
            if dep.dir not in done:
                docker.volume_rm(plan[dep.dir])
        if drop_cache:
            docker.volume_rm(cache)
    return plan
