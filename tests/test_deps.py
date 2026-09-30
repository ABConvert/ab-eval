from eval_harness.config import DepDir
from eval_harness.harness.deps import volume_name


def test_volume_name_is_docker_safe() -> None:
    assert (
        volume_name("demo-app", "web/frontend", "abc123def456")
        == "abeval-deps-demo-app-web_frontend-abc123def456"
    )


def test_one_preparation_at_a_time_per_volume_set() -> None:
    """Two cases sharing a lockfile must not install into the same volume at once."""
    import threading
    from types import SimpleNamespace

    from eval_harness.harness import deps

    ready: set[str] = set()
    overlapped = False
    inside = 0
    started = threading.Barrier(2, timeout=5)

    class FakeDocker:
        image = "img"

        def volume_rm(self, name: str) -> None:
            ready.discard(name)

        def volume_create(self, name: str) -> None:
            pass

        def volume_exists(self, name: str) -> bool:
            return True

        def rm(self, name: str) -> None:
            pass

        def create_container(self, **kw: object) -> str:
            return "cid"

        def start(self, cid: str) -> None:
            pass

        def cp_tar_in(self, cid: str, tar: bytes, dest: str) -> None:
            nonlocal overlapped, inside
            inside += 1
            overlapped = overlapped or inside > 1
            started.wait() if False else None
            import time

            time.sleep(0.05)
            inside -= 1

        def exec(self, cid: str, cmd: str, **kw: object) -> object:
            ready.add(volume)
            return SimpleNamespace(ok=True, stderr="")

    dep = DepDir(dir="web", lock=[], install="npm ci")
    repo = SimpleNamespace(key="r", image="img", dep_dirs=[dep], limits=None)
    volume = volume_name("r", "web", "d")

    monkey = deps._ready
    deps._ready = lambda docker, image, name: name in ready  # type: ignore[assignment]
    deps.lock_hash = lambda repo, dep, commit: "d"  # type: ignore[assignment]
    try:
        docker = FakeDocker()
        threads = [
            threading.Thread(
                target=deps.ensure_dep_volumes,
                args=(docker, repo, "c0ffee"),
                kwargs={"source_tar": b""},
            )
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
    finally:
        deps._ready = monkey  # type: ignore[assignment]

    assert not overlapped, "two threads prepared the same dependency volume at the same time"


def test_volumes_shared_across_different_plans_are_still_serialised() -> None:
    """Two cases differing on web but sharing web/frontend must not prepare it together."""
    import threading
    from types import SimpleNamespace

    from eval_harness.harness import deps

    ready: set[str] = set()
    overlapped = False
    inside = 0

    class FakeDocker:
        image = "img"

        def volume_rm(self, name: str) -> None:
            ready.discard(name)

        def volume_create(self, name: str) -> None:
            pass

        def volume_exists(self, name: str) -> bool:
            return True

        def rm(self, name: str) -> None:
            pass

        def create_container(self, **kw: object) -> str:
            return "cid"

        def start(self, cid: str) -> None:
            pass

        def cp_tar_in(self, cid: str, tar: bytes, dest: str) -> None:
            nonlocal overlapped, inside
            import time

            inside += 1
            overlapped = overlapped or inside > 1
            time.sleep(0.05)
            inside -= 1

        def exec(self, cid: str, cmd: str, **kw: object) -> object:
            return SimpleNamespace(ok=True, stderr="")

    web = DepDir(dir="web", lock=[], install="npm ci")
    fe = DepDir(dir="web/frontend", lock=[], install="npm ci")
    repo = SimpleNamespace(key="r", image="img", dep_dirs=[web, fe], limits=None)

    # Same frontend lockfile, different web lockfile: the plans differ, the frontend volume
    # does not.
    def lock_hash(repo: object, dep: object, commit: str) -> str:
        return commit if dep.dir == "web" else "shared"

    original_ready, original_hash = deps._ready, deps.lock_hash
    deps._ready = lambda docker, image, name: name in ready  # type: ignore[assignment]
    deps.lock_hash = lock_hash  # type: ignore[assignment]
    try:
        docker = FakeDocker()
        threads = [
            threading.Thread(
                target=deps.ensure_dep_volumes,
                args=(docker, repo, commit),
                kwargs={"source_tar": b""},
            )
            for commit in ("aaa", "bbb")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
    finally:
        deps._ready, deps.lock_hash = original_ready, original_hash  # type: ignore[assignment]

    assert not overlapped, "two cases prepared the shared web/frontend volume at the same time"


class RecordingDocker:
    """Records what prep asks of Docker, in order. `fail_on` makes that command fail."""

    def __init__(self, fail_on: str | None = None, timed_out: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.volumes: set[str] = set()
        self.mounts: dict[str, str] = {}
        self.fail_on = fail_on
        self.timed_out = timed_out

    def volume_exists(self, name: str) -> bool:
        return name in self.volumes

    def volume_create(self, name: str) -> None:
        self.calls.append(("volume_create", name))
        self.volumes.add(name)

    def volume_rm(self, name: str) -> None:
        self.calls.append(("volume_rm", name))
        self.volumes.discard(name)

    def rm(self, name: str) -> None:
        self.calls.append(("rm", name))

    def create_container(self, **kw: object) -> str:
        self.mounts = dict(kw["mounts"])  # type: ignore[call-overload]
        return "cid"

    def start(self, cid: str) -> None:
        pass

    def cp_tar_in(self, cid: str, tar: bytes, dest: str) -> None:
        pass

    def exec(self, cid: str, cmd: str, **kw: object) -> object:
        from types import SimpleNamespace

        self.calls.append(("exec", cmd, str(kw.get("cwd")), str(kw.get("timeout"))))
        failed = self.fail_on is not None and cmd == self.fail_on
        return SimpleNamespace(
            ok=not failed,
            stderr="boom",
            timed_out=failed and self.timed_out,
        )


def _repo(*deps: DepDir, runners: dict[str, object] | None = None) -> object:
    from eval_harness.config import RepoConfig

    return RepoConfig.model_validate(
        {
            "key": "r",
            "github": "acme/r",
            "local_path": "/nonexistent",
            "image": "img",
            "dockerfile": "docker/python312.Dockerfile",
            "dep_dirs": [d.model_dump() for d in deps],
            "runners": runners
            or {"py": {"kind": "pytest", "cwd": ".", "match": ["**/test_*.py"], "full": "pytest"}},
            "test_file_globs": ["**/tests/**"],
            "non_code_globs": [],
        }
    )


def _prepare(docker: RecordingDocker, repo: object, deps: list[DepDir]) -> dict[str, str]:
    from eval_harness.harness import deps as mod

    real_ready, real_hash = mod._ready, mod.lock_hash
    mod._ready = lambda docker, image, name: False  # type: ignore[assignment]
    mod.lock_hash = lambda repo, dep, commit: "d"  # type: ignore[assignment]
    try:
        return mod.ensure_dep_volumes(docker, repo, "c0ffee", source_tar=b"", dep_dirs=deps)  # type: ignore[arg-type]
    finally:
        mod._ready, mod.lock_hash = real_ready, real_hash  # type: ignore[assignment]


def test_a_uv_root_is_mounted_where_uv_installs() -> None:
    """A `uv sync` root must land on its volume: the old fixed node_modules mount lost it."""
    from eval_harness.harness.deps import mount_path, mounts_for

    venv = DepDir(dir=".", install="uv sync --frozen", lock=["uv.lock"], target=".venv")
    repo = _repo(venv)
    docker = RecordingDocker()
    plan = _prepare(docker, repo, [venv])

    assert mount_path(venv) == "/app/.venv"
    assert docker.mounts[plan["."]] == "/app/.venv"
    assert mounts_for(repo, plan) == {plan["."]: "/app/.venv"}  # type: ignore[arg-type]
    # The ready marker goes inside the mounted volume, or `_ready` never sees it.
    assert any(c[:3] == ("exec", "touch .abeval-ready", "/app/.venv") for c in docker.calls)


def test_npm_roots_keep_their_node_modules_mount() -> None:
    from eval_harness.harness.deps import mount_path

    assert mount_path(DepDir(dir="web", install="npm ci", lock=[])) == "/app/web/node_modules"


def test_prep_mounts_a_persistent_download_cache() -> None:
    from eval_harness.harness.deps import CACHE_MOUNT, cache_volume_name

    web = DepDir(dir="web", install="npm ci", lock=[])
    docker = RecordingDocker()
    _prepare(docker, _repo(web), [web])
    assert docker.mounts[cache_volume_name("r")] == CACHE_MOUNT


def test_install_timeout_comes_from_the_dep_dir() -> None:
    slow = DepDir(dir=".", install="uv sync", lock=[], target=".venv", install_timeout=7200)
    docker = RecordingDocker()
    _prepare(docker, _repo(slow), [slow])
    assert ("exec", "uv sync", "/app/.", "7200") in docker.calls


def test_a_failed_install_removes_every_unfinished_volume_after_the_container() -> None:
    """Removing a volume the prep container still mounts fails quietly and leaves it behind."""
    import pytest

    from eval_harness.harness.sandbox import DockerError

    web = DepDir(dir="web", install="npm ci", lock=[])
    fb = DepDir(dir="functions", install="npm ci --prefix functions", lock=[])
    docker = RecordingDocker(fail_on="npm ci", timed_out=True)
    with pytest.raises(DockerError, match="install_timeout"):
        _prepare(docker, _repo(web, fb), [web, fb])

    container_gone = docker.calls.index(("rm", "cid"))
    removed_after = {c[1] for c in docker.calls[container_gone:] if c[0] == "volume_rm"}
    assert removed_after == {"abeval-deps-r-web-d", "abeval-deps-r-functions-d"}


def test_runner_deps_narrow_what_a_case_prepares() -> None:
    import pytest

    web = DepDir(dir="web", install="npm ci", lock=[])
    py = DepDir(dir=".", install="uv sync", lock=[], target=".venv")
    runners = {
        "py": {"kind": "pytest", "cwd": ".", "match": ["**/*.py"], "full": "pytest", "deps": ["."]},
        "web": {"kind": "vitest", "cwd": "web", "match": ["web/**"], "full": "npx vitest run"},
    }
    repo = _repo(web, py, runners=runners)
    assert [d.dir for d in repo.deps_for([repo.runners["py"]])] == ["."]  # type: ignore[attr-defined]
    # A runner that does not say keeps the old behaviour: everything.
    assert [d.dir for d in repo.deps_for([repo.runners["web"]])] == ["web", "."]  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="not dep_dirs"):
        _repo(py, runners={"py": {**runners["py"], "deps": ["api"]}})
