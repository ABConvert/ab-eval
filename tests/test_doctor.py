"""`doctor` must be honest about what it found, and must not leak what it reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness import doctor, paths


def test_missing_data_root_fails_with_the_command_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "nope"))
    paths.reset_cache()
    check = doctor.check_data_root()[0]
    assert check.status == "fail"
    assert "mkdir" in check.fix


def test_empty_case_directory_warns_rather_than_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty data root is a fresh install, not a broken one."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    (tmp_path / "data" / "cases").mkdir(parents=True)
    by_name = {c.name: c for c in doctor.check_data_root()}
    assert by_name["data root"].status == "ok"
    assert by_name["cases"].status == "warn"
    assert "collect" in by_name["cases"].fix


def test_env_var_values_are_never_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check says whether a key is set. It must never say what it is."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value-do-not-print")
    checks = doctor.check_providers()
    blob = " ".join(f"{c.name} {c.detail} {c.fix}" for c in checks)
    assert "secret-value" not in blob
    assert any(c.name == "env ANTHROPIC_API_KEY" and c.detail == "set" for c in checks)


def test_worst_takes_the_most_severe(monkeypatch: pytest.MonkeyPatch) -> None:
    mk = lambda s: doctor.Check("x", s, "")  # noqa: E731
    assert doctor.worst([mk("ok"), mk("ok")]) == "ok"
    assert doctor.worst([mk("ok"), mk("warn")]) == "warn"
    assert doctor.worst([mk("warn"), mk("fail")]) == "fail"


def test_a_broken_check_does_not_hide_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[doctor.Check]:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(doctor, "check_docker", boom)
    checks = doctor.run_all()
    assert any(c.status == "fail" and "check itself failed" in c.detail for c in checks)
    assert any(c.name == "data root" for c in checks)


def test_setup_page_renders_every_check_and_no_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from starlette.testclient import TestClient

    from eval_harness.dashboard.app import build_app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-appear")
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    (tmp_path / "data" / "cases").mkdir(parents=True)

    body = TestClient(build_app()).get("/setup").text
    assert "must-not-appear" not in body
    assert "Setup" in body and "data root" in body


def test_setup_page_shows_the_config_it_reads_without_showing_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A page that names a file it is unhappy with should show you the file."""
    import shutil

    from starlette.testclient import TestClient

    from eval_harness.dashboard.app import build_app
    from tests.conftest import FIXTURES

    shutil.copytree(FIXTURES / "config", tmp_path / "config")
    (tmp_path / "data" / "cases").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    monkeypatch.setenv("DEMO_API_KEY", "sk-must-never-render")
    paths.reset_cache()

    body = TestClient(build_app()).get("/setup").text
    assert "demo-app" in body, "repos.yaml should be visible on the page"
    assert "DEMO_API_KEY" in body, "the variable's name is the useful part"
    assert "sk-must-never-render" not in body, "its value is not"


def test_checks_say_which_kind_of_thing_they_are() -> None:
    """Grouping matters: 'give the VM more memory' and 'edit a YAML' are different asks."""
    kinds = {c.kind for c in doctor.run_all()}
    assert kinds <= {"machine", "config", "secret", "data"}
    assert "config" in kinds


def test_setup_page_tells_a_fresh_clone_exactly_what_to_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The branches a stranger hits on their first run, which no other test reaches.

    A healthy machine never renders them, so a template error here is a 500 that only
    the new user ever sees — the worst possible audience for it.
    """
    from starlette.testclient import TestClient

    from eval_harness.dashboard.app import build_app

    fresh = [
        doctor.Check(
            "repos.yaml",
            "fail",
            f"not found at {tmp_path}/config/repos.yaml",
            "eval-harness init --repo <path to your repository>",
            kind="config",
        ),
        doctor.Check(
            "models.yaml",
            "fail",
            f"not found at {tmp_path}/config/models.yaml",
            f"cp config/models.example.yaml {tmp_path}/config/models.yaml",
            kind="config",
        ),
        doctor.Check(
            "cases",
            "warn",
            "0 case(s)",
            "eval-harness collect, or eval-harness import swebench",
            kind="data",
        ),
        doctor.Check(
            "env OPENAI_API_KEY",
            "warn",
            "not set",
            "export OPENAI_API_KEY=… in the shell that runs the harness",
            kind="secret",
        ),
    ]
    monkeypatch.setattr(doctor, "run_all", lambda: fresh)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-never-render")
    paths.reset_cache()

    # Bound to the network, so nothing is editable and every branch is the prose-and-command
    # one — the state this test was written for, now reachable only this way.
    body = TestClient(build_app("0.0.0.0")).get("/setup").text

    assert "Not ready to run" in body, "two failing checks is not a runnable machine"
    # The placeholder is quoted: pasted as-is it is a bad path, not a shell redirect.
    quoted = "eval-harness init --repo &#39;&lt;path to your repository&gt;&#39;"
    assert f'data-copy="{quoted}"' in body
    assert "eval-harness import swebench --limit 20" in body
    # The name and an empty value: the quote closes straight after the equals sign.
    assert 'data-copy="export OPENAI_API_KEY="' in body
    assert "sk-must-never-render" not in body

    # On loopback the same two config checks carry forms instead, because being told what to
    # type was the complaint. What cannot be done here is still a command, and says why.
    editable = TestClient(build_app()).get("/setup").text
    assert 'action="/setup/repos/detect"' in editable, "repos.yaml is drafted off the repository"
    assert 'action="/setup/models"' in editable, "models.yaml is written from the form"
    assert "eval-harness import swebench --limit 20" in editable, "collecting stays a command"
    assert 'data-copy="export OPENAI_API_KEY="' in editable, "and so does exporting a key"
    assert "a web page cannot export a variable into your shell" in editable
    # The one invariant both states share, and the only one that would be a breach.
    assert "sk-must-never-render" not in editable


def test_setup_page_never_offers_prose_as_a_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """`git rev-parse failed; is the checkout intact?` begins with a program name.

    Pasting it into a shell does nothing useful, so it has to render as a sentence.
    """
    from starlette.testclient import TestClient

    from eval_harness.dashboard.app import build_app

    prose = "git rev-parse failed; is the checkout intact?"
    monkeypatch.setattr(
        doctor,
        "run_all",
        lambda: [doctor.Check("repo demo", "fail", "/tmp/demo", prose)],
    )

    body = TestClient(build_app()).get("/setup").text
    assert f'data-copy="{prose}"' not in body, "prose must never get a copy button"
    assert "Git rev-parse failed; is the checkout intact?" in body


def _copyable_commands(body: str) -> list[str]:
    """Every string the page offers behind a Copy button."""
    import html
    import re

    return [html.unescape(m) for m in re.findall(r'data-copy="([^"]*)"', body)]


def test_no_copyable_command_is_a_shell_syntax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page promises "each one below says what to type". A command that errors the
    instant it is pasted breaks that for the one reader the page exists for.

    `<placeholder>` is the CLI's own --help convention and is fine in prose, but behind
    a Copy button bash reads it as a redirect.
    """
    import shutil
    import subprocess

    from starlette.testclient import TestClient

    from eval_harness.dashboard.app import build_app

    monkeypatch.setattr(
        doctor,
        "run_all",
        lambda: [
            doctor.Check(
                "repos.yaml",
                "fail",
                "not found at /tmp/x/config/repos.yaml",
                "eval-harness init --repo <path to your repository>",
                kind="config",
            ),
            doctor.Check("cases", "warn", "0 case(s)", "eval-harness collect", kind="data"),
            doctor.Check(
                "env OPENAI_API_KEY", "warn", "not set", "export OPENAI_API_KEY=…", kind="secret"
            ),
        ],
    )
    commands = _copyable_commands(TestClient(build_app()).get("/setup").text)
    assert commands, "the fresh-clone page must offer something to copy"

    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - every supported dev box has one
        pytest.skip("no bash to parse with")
    for cmd in commands:
        parsed = subprocess.run([bash, "-nc", cmd], capture_output=True, text=True)
        assert parsed.returncode == 0, f"{cmd!r} does not parse: {parsed.stderr.strip()}"


def test_the_example_this_page_offers_to_copy_actually_loads(tmp_path: Path) -> None:
    """/setup offers the shipped example as one click, so it has to load when clicked.

    It replaces a starter that used to live in the template as copyable text, where it drifted:
    Prices requires four floats (config.py) and that starter shipped two, so the first thing a
    stranger pasted raised a ValidationError. One example, checked here, is the fix for that
    class of drift — there is now nowhere else for a second copy to disagree from.
    """
    from starlette.testclient import TestClient

    from eval_harness import config
    from eval_harness.dashboard.app import build_app

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(paths.ENV_VAR, str(tmp_path))
        paths.reset_cache()
        r = TestClient(build_app()).post("/setup/models-from-example", follow_redirects=False)
        assert r.status_code == 303, r.text
        models = config.load_models(tmp_path / "config" / "models.yaml")
    finally:
        monkey.undo()
        paths.reset_cache()

    assert "judge" in models, "an example without a judge cannot score a run"
    runnable = [k for k in models if k not in ("judge", "classifier")]
    assert runnable, "and one with nothing to put on trial is not a starting point"
    # Any entry that names a price has to name all four, or the entry does not load at all.
    assert any(m.price_per_mtok is not None for m in models.values())


def _one_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, linear_team: str) -> None:
    """A data root holding one repository, whose ticket source is what is under test."""
    from tests.conftest import FIXTURES

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    for name in ("models.yaml", "repos.yaml"):
        (cfg / name).write_text((FIXTURES / "config" / name).read_text())
    repos = (cfg / "repos.yaml").read_text()
    assert "linear_team: DEMO" in repos, "the fixture the substitution below depends on"
    line = f"linear_team: {linear_team}" if linear_team else 'linear_team: ""'
    (cfg / "repos.yaml").write_text(repos.replace("linear_team: DEMO", line))
    (tmp_path / "data" / "cases").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()


def test_a_linear_repo_without_the_key_fails_and_says_where_one_comes_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Setting linear_team is the act that needs the key, and nothing used to mention it."""
    _one_repo(tmp_path, monkeypatch, linear_team="DEMO")
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)

    check = {c.name: c for c in doctor.check_tickets()}["tickets demo-app"]
    assert check.status == "fail" and check.kind == "secret"
    assert "DEMO" in check.detail and "not set" in check.detail
    assert check.fix.startswith("export LINEAR_API_KEY=")
    assert "Personal API keys" in check.fix, "a key nobody can find is not a fix"


def test_a_linear_repo_with_the_key_passes_without_claiming_it_works(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Set is the whole claim. Nothing here calls Linear, so nothing here may imply it did."""
    _one_repo(tmp_path, monkeypatch, linear_team="DEMO")
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_must-not-appear")

    checks = doctor.check_tickets()
    check = {c.name: c for c in checks}["tickets demo-app"]
    assert check.status == "ok" and check.fix == ""
    assert "not checked against Linear" in check.detail
    blob = " ".join(f"{c.name} {c.detail} {c.fix}" for c in checks)
    assert "must-not-appear" not in blob


def test_a_github_issues_repo_is_never_asked_for_a_linear_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _one_repo(tmp_path, monkeypatch, linear_team="")
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)

    checks = doctor.check_tickets()
    check = {c.name: c for c in checks}["tickets demo-app"]
    assert check.status == "ok"
    assert "GitHub Issues" in check.detail, "which source it is, said out loud"
    blob = " ".join(f"{c.name} {c.detail} {c.fix}" for c in checks)
    assert "LINEAR_API_KEY" not in blob, "a repository that never calls Linear needs no key"


def test_no_repositories_means_no_ticket_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """repos.yaml owns that state. Two checks for one missing file is a worse worklist."""
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path / "checkout")
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "root"))
    paths.reset_cache()
    assert doctor.check_tickets() == []


def test_a_missing_gh_is_a_warning_on_every_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Collect reads the pull requests through gh whichever tracker holds the tickets.

    Presence only, deliberately: `gh auth status` is a round trip to GitHub on every render of
    the page, and an unauthenticated gh already fails with its own message — a missing one is a
    bare FileNotFoundError, which is the silence this check exists for.
    """
    _one_repo(tmp_path, monkeypatch, linear_team="DEMO")
    real = doctor.shutil.which
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "gh" else real(n))

    check = {c.name: c for c in doctor.check_tickets()}["gh"]
    assert check.status == "warn" and check.kind == "machine"
    assert "gh auth login" in check.fix


def test_the_linear_key_is_never_rendered_on_the_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The name is the useful part, on this check as on every other."""
    from starlette.testclient import TestClient

    from eval_harness.dashboard.app import build_app

    _one_repo(tmp_path, monkeypatch, linear_team="DEMO")
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_must-not-appear")

    body = TestClient(build_app()).get("/setup").text
    assert "must-not-appear" not in body
    assert "LINEAR_API_KEY" in body, "the variable is named; the value never is"
    assert "setting it reads tickets from Linear" in body, "said where the switch is thrown"
