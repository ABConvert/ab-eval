"""Real Docker proof that writable attempt dependencies cannot poison a later attempt."""

import uuid
from pathlib import Path

import pytest

from eval_harness.config import load_repos
from eval_harness.harness.deps import isolated_dep_volumes, mounts_for
from eval_harness.harness.sandbox import Docker

pytestmark = pytest.mark.docker


def test_dependency_changes_do_not_survive_an_attempt() -> None:
    docker = Docker()
    repo = load_repos(Path(__file__).parents[1] / "fixtures/config/repos.yaml")["demo-app"]
    repo = repo.model_copy(update={"image": "alpine:latest"})
    if not docker.image_exists(repo.image):
        pytest.skip("requires a local alpine:latest image")
    identity = uuid.uuid4().hex
    cache, seed = f"abeval-test-cache-{identity}", f"abeval-test-seed-{identity}"
    copies: list[str] = []
    docker.volume_create(cache)
    try:
        docker.create_container(
            image=repo.image,
            name=seed,
            mounts={cache: "/seed"},
            limits=repo.limits,
            network=False,
            workdir="/",
        )
        docker.start(seed)
        docker.write_file(seed, "/seed/value", "trusted")
        docker.rm(seed)
        for index in range(2):
            with isolated_dep_volumes(docker, repo, {"web": cache}) as plan:
                copies.extend(plan.values())
                attempt = f"abeval-test-attempt-{identity}-{index}"
                try:
                    docker.create_container(
                        image=repo.image,
                        name=attempt,
                        mounts=mounts_for(repo, plan),
                        limits=repo.limits,
                        network=False,
                    )
                    docker.start(attempt)
                    assert docker.read_file(attempt, "/app/web/node_modules/value") == "trusted"
                    docker.write_file(attempt, "/app/web/node_modules/value", "model modification")
                    assert (
                        docker.read_file(attempt, "/app/web/node_modules/value")
                        == "model modification"
                    )
                finally:
                    docker.rm(attempt)
            assert all(not docker.volume_exists(volume) for volume in copies)
        assert len(set(copies)) == 2
    finally:
        docker.rm(seed)
        docker.volume_rm(cache)
