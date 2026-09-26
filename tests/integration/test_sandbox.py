import pytest

from eval_harness import PROJECT_ROOT
from eval_harness.collect.cases import load_case
from eval_harness.config import load_repos
from eval_harness.harness.deps import ensure_dep_volumes
from eval_harness.harness.sandbox import Docker
from eval_harness.harness.source import (
    apply_patch,
    archive_tar,
    diff_head,
    git_init,
    populate_source,
)
from eval_harness.harness.testrun import run_case_tests

pytestmark = pytest.mark.docker


def test_pro2877_source_deps_and_tests() -> None:
    repo = load_repos()["demo-app"]
    case = load_case("DEMO-2877")
    docker = Docker()
    assert docker.daemon_ok()
    if not docker.image_exists(repo.image):
        docker.build_image(repo.image, PROJECT_ROOT / repo.dockerfile, PROJECT_ROOT)
    tar = archive_tar(repo, case.base_commit)
    volumes = ensure_dep_volumes(docker, repo, case.base_commit, source_tar=tar)
    docker.rm("abeval-test-pro2877")
    cid = docker.create_container(
        image=repo.image,
        name="abeval-test-pro2877",
        mounts={v: f"/app/{d}/node_modules" for d, v in volumes.items()},
        limits=repo.limits,
        network=False,
    )
    try:
        docker.start(cid)
        populate_source(docker, cid, tar)
        base = git_init(docker, cid)
        assert len(base) == 40
        assert docker.exec(cid, "git log --oneline | wc -l").stdout.strip() == "1"
        assert docker.exec(cid, "test -f web/node_modules/.abeval-ready").ok
        mongod = docker.exec(cid, "ls web/node_modules/.cache/mongodb-memory-server").stdout
        assert "mongod" in mongod
        assert not docker.exec(cid, "curl -sS --max-time 3 https://registry.npmjs.org/").ok
        apply_patch(docker, cid, case.human_test_patch, "tests")
        assert docker.exec(cid, "git log --oneline | wc -l").stdout.strip() == "2"
        assert diff_head(docker, cid) == ""
        runner = repo.runners[case.test_runner]
        before = run_case_tests(docker, cid, runner, case.test_files)
        assert before.total > 0 and before.failed >= 1, before.output[-2000:]
        apply_patch(docker, cid, case.human_patch, "human")
        after = run_case_tests(docker, cid, runner, case.test_files)
        assert after.ok, after.output[-2000:]
    finally:
        docker.rm(cid)
