from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from eval_harness.config import load_repos
from eval_harness.harness.sandbox import Docker, DockerError, ExecResult


class VolumeDocker(Docker):
    """Model Docker volume contents so mutations survive container removal, as in Docker."""

    def __init__(self, fail: bool = False) -> None:
        self.volumes = {"cache": {"package.py": "trusted"}}
        self.containers: dict[str, dict[str, str]] = {}
        self.fail = fail

    def volume_create(self, name: str) -> None:
        self.volumes[name] = {}

    def volume_rm(self, name: str) -> None:
        assert all(name not in mounts for mounts in self.containers.values())
        del self.volumes[name]

    def create_container(self, **kwargs: Any) -> str:
        assert kwargs["network"] is False
        self.containers[kwargs["name"]] = kwargs["mounts"]
        return str(kwargs["name"])

    def start(self, cid: str) -> None:
        pass

    def exec(self, cid: str, cmd: Any, **kwargs: Any) -> ExecResult:
        mounts = self.containers[cid]
        assert mounts["cache"].endswith(":ro")
        if self.fail:
            return ExecResult(1, "", "synthetic copy failure")
        target = next(v for v in mounts if v != "cache")
        self.volumes[target] = dict(self.volumes["cache"])
        return ExecResult(0, "", "")

    def rm(self, cid: str) -> None:
        self.containers.pop(cid, None)


@pytest.fixture
def isolation() -> Iterator[Any]:
    from eval_harness.harness.deps import isolated_dep_volumes

    yield isolated_dep_volumes


def test_one_attempt_cannot_poison_the_next(isolation: Any) -> None:
    repo = load_repos(Path(__file__).parent / "fixtures/config/repos.yaml")["demo-app"]
    docker = VolumeDocker()
    with isolation(docker, repo, {"web": "cache"}) as first:
        docker.volumes[first["web"]]["package.py"] = "model modification"
        assert docker.volumes["cache"]["package.py"] == "trusted"
        with isolation(docker, repo, {"web": "cache"}) as second:
            assert first["web"] != second["web"]
            assert docker.volumes[second["web"]]["package.py"] == "trusted"
    assert docker.volumes == {"cache": {"package.py": "trusted"}}
    assert docker.containers == {}


@pytest.mark.parametrize("during_copy", [False, True])
def test_attempt_volumes_are_removed_on_failure(isolation: Any, during_copy: bool) -> None:
    repo = load_repos(Path(__file__).parent / "fixtures/config/repos.yaml")["demo-app"]
    docker = VolumeDocker(fail=during_copy)
    with pytest.raises((DockerError, RuntimeError)), isolation(docker, repo, {"web": "cache"}):
        raise RuntimeError("synthetic attempt failure")
    assert docker.volumes == {"cache": {"package.py": "trusted"}}
    assert docker.containers == {}


async def test_runner_removes_attempt_copies_on_early_validation_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eval_harness import paths
    from eval_harness.adapters.base import Caps
    from eval_harness.adapters.fake import NoopAdapter
    from eval_harness.harness import runner
    from eval_harness.harness.testrun import TestResult
    from tests.test_runner import _case

    repo = load_repos(Path(__file__).parent / "fixtures/config/repos.yaml")["demo-app"]
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    monkeypatch.setattr(runner, "archive_tar", lambda *args: b"")
    monkeypatch.setattr(runner, "ensure_dep_volumes", lambda *args, **kw: {"web": "cache"})
    monkeypatch.setattr(runner, "populate_source", lambda *args: None)
    monkeypatch.setattr(runner, "git_init", lambda *args: "base")
    monkeypatch.setattr(runner, "apply_patch", lambda *args: "tests")
    monkeypatch.setattr(
        runner,
        "run_groups",
        lambda *args, **kw: TestResult(
            total=1, passed=1, failed=0, skipped=0, exit_code=0, output=""
        ),
    )
    docker = VolumeDocker()
    mounts: list[dict[str, str]] = []
    create = docker.create_container

    def record_create(**kw: Any) -> str:
        mounts.append(dict(kw["mounts"]))
        return create(**kw)

    monkeypatch.setattr(docker, "create_container", record_create)
    rec = await runner.run_case(
        _case("C1"),
        repo,
        adapter=NoopAdapter(),
        caps=Caps(),
        run_id="validate",
        docker=docker,
        validate_only=True,
    )
    assert rec.status == "invalid", rec.error  # Already-passing tests take an early return.
    assert len(mounts) == 2
    assert "cache" not in mounts[-1]
    assert docker.containers == {}
    assert docker.volumes == {"cache": {"package.py": "trusted"}}


async def test_cancelling_during_copy_waits_for_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import threading
    from contextlib import contextmanager

    from eval_harness import paths
    from eval_harness.adapters.base import Caps
    from eval_harness.adapters.fake import NoopAdapter
    from eval_harness.harness import runner
    from tests.test_runner import _case

    repo = load_repos(Path(__file__).parent / "fixtures/config/repos.yaml")["demo-app"]
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    @contextmanager
    def copying(*args: Any) -> Iterator[dict[str, str]]:
        entered.set()
        assert release.wait(5)
        try:
            yield {"web": "copy"}
        finally:
            closed.set()

    monkeypatch.setattr(runner, "archive_tar", lambda *args: b"")
    monkeypatch.setattr(runner, "ensure_dep_volumes", lambda *args, **kw: {"web": "cache"})
    monkeypatch.setattr(runner, "isolated_dep_volumes", copying)
    task = asyncio.create_task(
        runner.run_case(
            _case("C1"),
            repo,
            adapter=NoopAdapter(),
            caps=Caps(),
            run_id="validate",
            docker=VolumeDocker(),
            validate_only=True,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "cleanup must wait until the copying thread returns"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()
