"""`init` reads a repository; these build small ones and check what it concludes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from eval_harness import scaffold
from eval_harness.config import RepoConfig


def _git_repo(root: Path, remote: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    # Test commits must not depend on the contributor's global identity or signing setup.
    for key, value in (
        ("user.name", "Test User"),
        ("user.email", "test@example.invalid"),
        ("commit.gpgSign", "false"),
    ):
        subprocess.run(["git", "-C", str(root), "config", key, value], check=True)
    if remote:
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", remote], check=True)
    return root


def _node_project(root: Path, rel: str, *, lock: str = "package-lock.json") -> Path:
    d = root / rel if rel != "." else root
    d.mkdir(parents=True, exist_ok=True)
    (d / "package.json").write_text(json.dumps({"name": rel, "engines": {"node": ">=22"}}))
    (d / lock).write_text("{}")
    return d


def test_remote_is_read_from_either_url_form(tmp_path: Path) -> None:
    ssh = _git_repo(tmp_path / "a", "git@github.com:acme/widget.git")
    https = _git_repo(tmp_path / "b", "https://github.com/acme/widget")
    assert scaffold.detect_remote(ssh) == "acme/widget"
    assert scaffold.detect_remote(https) == "acme/widget"
    assert scaffold.detect_remote(_git_repo(tmp_path / "c")) is None


def test_dep_dirs_need_a_lockfile_beside_the_manifest(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    _node_project(repo, "web")
    (repo / "docs").mkdir()
    (repo / "docs" / "package.json").write_text("{}")  # no lockfile: not a dependency root
    dirs = {d.dir: d for d in scaffold.detect_dep_dirs(repo)}
    assert set(dirs) == {"web"}
    assert dirs["web"].install == "npm ci"
    assert dirs["web"].lock == ["web/package.json", "web/package-lock.json"]


def test_lockfile_chooses_the_install_command(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    _node_project(repo, "app", lock="pnpm-lock.yaml")
    assert scaffold.detect_dep_dirs(repo)[0].install == "pnpm install --frozen-lockfile"


def test_vendored_trees_are_skipped(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    _node_project(repo, "node_modules/sneaky")
    assert scaffold.detect_dep_dirs(repo) == []


def test_runner_globs_come_from_the_test_files_that_exist(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    web = _node_project(repo, "web")
    (web / "vitest.config.ts").write_text("export default {}")
    (web / "a.test.ts").write_text("")
    (web / "b.test.tsx").write_text("")
    runner = scaffold.detect_runners(repo)[0]
    assert runner.kind == "vitest"
    assert runner.cwd == "web"
    assert runner.match == ["web/**/*.test.ts", "web/**/*.test.tsx"]


def test_jest_is_reported_not_mislabelled(tmp_path: Path) -> None:
    """There is no jest parser, so emitting one as vitest would fail confusingly."""
    repo = _git_repo(tmp_path / "repo")
    web = _node_project(repo, "web")
    (web / "jest.config.js").write_text("module.exports = {}")
    (web / "a.test.js").write_text("")
    assert scaffold.detect_runners(repo) == []
    assert scaffold.detect_jest_dirs(repo) == ["web"]
    assert any("jest" in n.lower() for n in scaffold.detect(repo).notes)


def test_node_version_comes_from_nvmrc_before_engines(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    _node_project(repo, ".")
    assert scaffold.detect_base_image(repo)[0].endswith(":node22")
    (repo / ".nvmrc").write_text("18.19.0\n")
    assert scaffold.detect_base_image(repo)[0].endswith(":node18")


def test_ticket_prefix_needs_a_dominant_one(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    for i in range(6):
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", f"ENG-{i}: work"],
            check=True,
        )
    assert scaffold.detect_ticket_prefix(repo) == "ENG"


def test_one_off_prefixes_are_not_a_convention(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "ABC-1: once"], check=True
    )
    assert scaffold.detect_ticket_prefix(repo) is None


def test_rendered_yaml_is_accepted_by_the_config_model(tmp_path: Path) -> None:
    """The whole point: what init writes must be what the harness can load."""
    repo = _git_repo(tmp_path / "repo", "git@github.com:acme/widget.git")
    web = _node_project(repo, "web")
    (web / "vitest.config.ts").write_text("export default {}")
    (web / "a.test.ts").write_text("")

    raw = yaml.safe_load(scaffold.render(scaffold.detect(repo)))
    key = next(iter(raw))
    body = dict(raw[key]) | {"key": key, "local_path": Path(raw[key]["local_path"])}
    cfg = RepoConfig.model_validate(body)

    assert cfg.github == "acme/widget"
    assert cfg.linear_team == ""  # no prefix found, so tickets come from GitHub Issues
    assert list(cfg.runners) == ["web"]


def test_every_guess_is_marked_for_checking(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    body = scaffold.render(scaffold.detect(repo))
    assert body.count("CHECK") >= 3


def test_detection_prunes_vendored_trees_instead_of_walking_them(tmp_path: Path) -> None:
    """A repository with node_modules holds hundreds of thousands of files.

    Walking them all took two minutes on a real monorepo and made `init` unusable. The walk
    prunes, so a deep vendored tree costs nothing.
    """
    repo = _git_repo(tmp_path / "repo")
    web = _node_project(repo, "web")
    (web / "vitest.config.ts").write_text("export default {}")
    (web / "a.test.ts").write_text("")

    buried = web / "node_modules" / "pkg" / "deep" / "deeper"
    buried.mkdir(parents=True)
    for i in range(300):
        (buried / f"m{i}.test.ts").write_text("")
        (buried / f"p{i}.json").write_text("{}")

    runner = scaffold.detect_runners(repo)[0]
    assert runner.match == ["web/**/*.test.ts"], "vendored test files must not shape the globs"
    assert scaffold.detect_dep_dirs(repo) == [
        d for d in scaffold.detect_dep_dirs(repo) if "node_modules" not in d.dir
    ]


def _py_project(root: Path, rel: str = ".") -> Path:
    d = root / rel if rel != "." else root
    d.mkdir(parents=True, exist_ok=True)
    (d / "pyproject.toml").write_text('[project]\nname = "x"\n[tool.pytest.ini_options]\n')
    (d / "uv.lock").write_text("")
    return d


def test_nested_ignored_checkouts_are_not_the_project(tmp_path: Path) -> None:
    """A clone holding its own worktrees yielded one dep root and runner per copy."""
    repo = _git_repo(tmp_path / "r")
    _node_project(repo, "web")
    (repo / ".gitignore").write_text(".worktrees/\n")
    for i in range(3):
        _node_project(repo, f".worktrees/copy{i}/web")
    assert [d.dir for d in scaffold.detect_dep_dirs(repo)] == ["web"]


def test_the_image_is_named_after_the_key_in_lowercase(tmp_path: Path) -> None:
    """A checkout folder called NexRex-Eval gave `abeval/NexRex-Eval:…`, which Docker refuses."""
    repo = _git_repo(tmp_path / "Some-Checkout")
    _py_project(repo)
    det = scaffold.detect(repo, key="my-app")
    assert det.image == "abeval/my-app:python312"
    assert scaffold.detect(repo).image == "abeval/some-checkout:python312"


def test_a_python_and_node_repository_gets_an_image_with_both(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "r")
    _py_project(repo)
    _node_project(repo, "web")
    det = scaffold.detect(repo, key="mixed")
    assert det.dockerfile == "docker/python312-node20.Dockerfile"
    assert det.image == "abeval/mixed:python312-node20"


def test_a_uv_root_targets_its_venv_and_each_runner_names_its_deps(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "r")
    _py_project(repo)
    web = _node_project(repo, "web")
    (web / "vitest.config.ts").write_text("")
    (web / "a.test.ts").write_text("")
    det = scaffold.detect(repo, key="mixed")
    venv = next(d for d in det.dep_dirs if d.dir == ".")
    assert venv.target == ".venv"
    deps = {r.kind: r.deps for r in det.runners}
    assert deps == {"vitest": ["web"], "pytest": ["."]}

    cfg = RepoConfig.model_validate(
        {"key": "mixed", **yaml.safe_load(scaffold.render(det))["mixed"]}
    )
    assert cfg.deps_for([cfg.runners["pytest"]])[0].target == ".venv"


def test_install_flags_and_test_env_are_read_from_ci(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "r")
    _py_project(repo)
    _node_project(repo, "web")
    wf = repo / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text(
        "jobs:\n  t:\n    steps:\n"
        "      - run: cd web && npm ci --legacy-peer-deps\n"
        "      - run: APPLICATION_ENV=test uv run pytest -q\n"
    )
    det = scaffold.detect(repo, key="ci")
    assert next(d for d in det.dep_dirs if d.dir == "web").install == "npm ci --legacy-peer-deps"
    assert next(r for r in det.runners if r.kind == "pytest").env == {"APPLICATION_ENV": "test"}


def test_two_lockfiles_install_the_way_ci_does(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "r")
    web = _node_project(repo, "web")
    (web / "pnpm-lock.yaml").write_text("")
    assert scaffold.detect_dep_dirs(repo)[0].install == "npm ci"
    ci = "steps:\n  - run: pnpm install --frozen-lockfile\n"
    assert scaffold.detect_dep_dirs(repo, ci)[0].install.startswith("pnpm install")
