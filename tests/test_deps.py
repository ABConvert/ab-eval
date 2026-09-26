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

    dep = SimpleNamespace(dir="web", lock=[], env={}, install="npm ci", post_install=None)
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

    web = SimpleNamespace(dir="web", lock=[], env={}, install="npm ci", post_install=None)
    fe = SimpleNamespace(dir="web/frontend", lock=[], env={}, install="npm ci", post_install=None)
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
