"""/setup writes config, and these are the ways it must refuse to.

Every test here relocates the data root before it touches anything. `conftest` points it at
`tests/fixtures`, so a write test that forgot would edit the fixture config every other test
reads — and the last test in this file asserts that none of them did.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eval_harness import paths
from eval_harness.config import load_models, load_repos
from eval_harness.dashboard import config_io
from eval_harness.dashboard.app import build_app
from tests.conftest import FIXTURES
from tests.dashboard_client import TestClient

# What tests/fixtures/config held when this module was imported, checked again at the end.
FIXTURE_DIGESTS = {
    p.name: hashlib.md5(p.read_bytes()).hexdigest()
    for p in sorted((FIXTURES / "config").glob("*.yaml"))
}

A_MODEL = {
    "key": "demo-api",
    "provider": "anthropic",
    "model": "demo-model-2",
    "effort": "medium",
    "max_output_tokens": "9000",
}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data root with its own copy of the fixture config, safe to write over."""
    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    for name in ("models.yaml", "repos.yaml"):
        (cfg / name).write_text((FIXTURES / "config" / name).read_text())
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    return cfg


@pytest.fixture
def fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh clone: no config on the data root and none on the checkout either.

    Both halves matter. `config_file` falls back to the checkout when the data root has no
    copy of its own, so a test that only emptied the data root would still be reading this
    repository's real config — and would pass while the page silently had something to show.
    """
    checkout = tmp_path / "checkout"
    (checkout / "config").mkdir(parents=True)
    monkeypatch.setattr(paths, "PROJECT_ROOT", checkout)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "root"))
    paths.reset_cache()
    assert not paths.config_file("models.yaml").exists(), "the state under test"
    return tmp_path / "root" / "config"


def test_a_valid_save_round_trips_through_the_loader(configured: Path) -> None:
    """The point of the page: an edit in the browser is an edit the harness then reads."""
    before = load_models(configured / "models.yaml")["demo-api"]
    assert before.model == "demo-model-1", "the state under test"

    client = TestClient(build_app())
    r = client.post("/setup/models/demo-api", data=A_MODEL, follow_redirects=False)
    assert r.status_code == 303, r.text
    assert "/setup?saved=" in r.headers["location"], "and it says what it did"

    after = load_models(configured / "models.yaml")
    assert after["demo-api"].model == "demo-model-2"
    assert after["demo-api"].effort == "medium"
    assert after["demo-api"].max_output_tokens == 9000
    # Everything the form did not carry is still there. A save that quietly dropped the judge
    # would leave `eval-harness review` broken and nothing on screen to say why.
    assert set(after) == {"demo-api", "demo-compat", "judge", "classifier"}
    assert after["judge"].model == "demo-judge-1"


def test_a_save_that_does_not_validate_writes_nothing(configured: Path) -> None:
    """The whole reason to validate before writing: a config the harness cannot load is worse
    than no editor, because it breaks an install that was working."""
    models = configured / "models.yaml"
    before = _md5(models)

    client = TestClient(build_app())
    r = client.post(
        "/setup/config/models.yaml/raw",
        data={
            "body": "demo-api:\n  provider: anthropic\n  model: x\n"
            "  price_per_mtok: {input: 1, output: 2}\n"
        },
    )
    assert r.status_code == 422
    assert _md5(models) == before, "the file on disk is untouched"
    assert "cache_read" in r.text, "and the loader's own words reach the page"
    assert "Field required" in r.text
    assert "demo-api:" in r.text, "with what was typed still in the editor"


def test_a_key_pasted_into_api_key_env_is_refused(configured: Path) -> None:
    """api_key_env is typed `str | None`, so pydantic accepts a credential there happily.

    Both routes reach that field and both have to refuse. The form is the one a person uses;
    the YAML editor is the one that would otherwise be the way around it.
    """
    models = configured / "models.yaml"
    before = _md5(models)
    client = TestClient(build_app())

    r = client.post(
        "/setup/models/demo-api",
        data={**A_MODEL, "api_key_env": "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
    )
    assert r.status_code == 422
    assert _md5(models) == before
    assert "looks like an API key" in r.text and "committed to version control" in r.text
    assert "sk-ant-api03-AAAA" not in r.text, "and the page never echoes the key back"

    r = client.post(
        "/setup/config/models.yaml/raw",
        data={
            "body": "demo-api:\n  provider: anthropic\n  model: x\n"
            "  api_key_env: sk-proj-BBBBBBBBBBBBBBBBBBBBBBBB\n"
        },
    )
    assert r.status_code == 422
    assert _md5(models) == before

    # A name that is merely not a variable name is refused too, with the other message —
    # this one validates cleanly and would have been written.
    r = client.post(
        "/setup/config/models.yaml/raw",
        data={"body": "demo-api:\n  provider: anthropic\n  model: x\n  api_key_env: my-key\n"},
    )
    assert r.status_code == 422
    assert _md5(models) == before
    assert "environment variable name" in r.text


def test_a_dashboard_off_loopback_refuses_to_start(configured: Path) -> None:
    before = _md5(configured / "models.yaml")
    with pytest.raises(ValueError, match="loopback"):
        build_app("0.0.0.0")
    assert _md5(configured / "models.yaml") == before


def test_a_cross_site_post_is_refused_even_on_loopback(configured: Path) -> None:
    """Loopback keeps other machines out; it does not keep out a page the reader is visiting,
    whose form can post here from their own browser."""
    models = configured / "models.yaml"
    before = _md5(models)
    client = TestClient(build_app())
    r = client.post(
        "/setup/models/demo-api", data=A_MODEL, headers={"sec-fetch-site": "cross-site"}
    )
    assert r.status_code == 403
    assert _md5(models) == before
    assert "arrived from another site" in r.text
    assert client.post("/setup/models/demo-api", data=A_MODEL).status_code in (303, 200)


def test_arriving_from_a_link_does_not_make_the_page_read_only(configured: Path) -> None:
    """Chrome reports `cross-site` on any top-level navigation the reader did not type — a
    bookmark from another site, a link in a chat, a README.

    That was folded into the editable decision at first, so following a link to the dashboard
    silently produced the read-only page, on loopback, with a banner blaming the bind address
    it was not bound to. Where the reader clicked from says nothing about whether this
    installation may be edited; it only says whether a particular write was forged.
    """
    client = TestClient(build_app())
    page = client.get(
        "/setup", headers={"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate"}
    )
    assert page.status_code == 200
    assert "This page can write your configuration" in page.text
    assert "Read-only while the harness is reachable" not in page.text


def test_the_previous_version_is_kept_before_it_is_replaced(configured: Path) -> None:
    """A stray command has emptied this file before. The backup costs kilobytes."""
    models = configured / "models.yaml"
    original = models.read_text()

    client = TestClient(build_app())
    r = client.post("/setup/models/demo-api", data=A_MODEL, follow_redirects=False)
    assert r.status_code == 303

    saved = list(configured.glob("models.yaml.*.bak"))
    assert len(saved) == 1, saved
    assert saved[0].read_text() == original, "byte for byte what was replaced"
    assert not saved[0].name.endswith(".yaml"), "a backup that looks like config gets loaded"
    assert saved[0].name in r.headers["location"], "and the page says where it went"


def test_a_fresh_clone_can_write_both_files_from_nothing(fresh: Path) -> None:
    """The state a stranger arrives in. Neither file exists and the page has to create them."""
    client = TestClient(build_app())
    page = client.get("/setup")
    assert page.status_code == 200
    assert "Read the repository" in page.text, "repos.yaml is drafted off the repository"
    assert "Add model" in client.get("/setup?step=model").text

    r = client.post(
        "/setup/models",
        data={
            "key": "claude-sonnet-5",
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "effort": "high",
            "max_output_tokens": "16000",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    assert (fresh / "models.yaml").exists()
    # Written to the data root, never through config_file's fallback to the checkout.
    assert not (paths.PROJECT_ROOT / "config" / "models.yaml").exists()
    assert load_models(fresh / "models.yaml")["claude-sonnet-5"].model == "claude-sonnet-5"

    r = client.post(
        "/setup/config/repos.yaml/raw",
        data={"body": (FIXTURES / "config" / "repos.yaml").read_text()},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    assert load_repos(fresh / "repos.yaml")["demo-app"].github == "acme/demo-app"
    assert (fresh / "repos.yaml").read_text().startswith("#"), "the YAML path keeps comments"

    # Nothing to back up on a first write, so nothing claims there was.
    assert not list(fresh.glob("*.bak"))


def test_the_repo_form_leaves_everything_it_does_not_cover_alone(configured: Path) -> None:
    """The form is half the file on purpose. The other half has to survive a save from it."""
    client = TestClient(build_app())
    r = client.post(
        "/setup/repos/demo-app",
        data={
            "github": "acme/demo-app",
            "local_path": "/tmp/moved",
            "linear_team": "DEMO",
            "image": "abeval/demo-app:node20",
            "dockerfile": "docker/node20.Dockerfile",
            "cpus": "4",
            "memory": "8g",
            "pids": "4096",
            "agent_authors": "demo-bot: agent\ndemo-unclear: unclear",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text

    repo = load_repos(configured / "repos.yaml")["demo-app"]
    assert str(repo.local_path) == "/tmp/moved"
    assert repo.limits.memory == "8g" and repo.limits.pids == 4096
    assert repo.agent_authors == {"demo-bot": "agent", "demo-unclear": "unclear"}
    # None of this was on the form, and all of it is what makes the repository runnable.
    assert list(repo.runners) == ["integration", "frontend", "script", "unit"]
    assert repo.runners["unit"].full == "npx vitest run --config vitest.unit.config.ts"
    assert len(repo.dep_dirs) == 3
    assert len(repo.test_file_globs) == 5


def test_renaming_a_model_does_not_leave_the_old_name_behind(configured: Path) -> None:
    client = TestClient(build_app())
    r = client.post(
        "/setup/models/demo-api", data={**A_MODEL, "key": "renamed"}, follow_redirects=False
    )
    assert r.status_code == 303
    models = load_models(configured / "models.yaml")
    assert "renamed" in models and "demo-api" not in models


def test_partial_prices_are_refused_with_the_reason(configured: Path) -> None:
    """Pydantic would say `Field required`; the page can say why all four exist."""
    before = _md5(configured / "models.yaml")
    client = TestClient(build_app())
    r = client.post("/setup/models/demo-api", data={**A_MODEL, "price_input": "2"})
    assert r.status_code == 422
    assert _md5(configured / "models.yaml") == before
    assert "all four or none" in r.text
    assert "cache_read" in r.text, "and which ones are still empty"


def test_a_repeated_key_is_caught_before_one_of_them_is_lost(configured: Path) -> None:
    """yaml.safe_load keeps the last of a repeated key and says nothing, which is exactly what
    pasting a drafted entry under an existing one produces."""
    before = _md5(configured / "models.yaml")
    client = TestClient(build_app())
    r = client.post(
        "/setup/config/models.yaml/raw",
        data={
            "body": "judge:\n  provider: anthropic\n  model: a\n"
            "judge:\n  provider: anthropic\n  model: b\n"
        },
    )
    assert r.status_code == 422
    assert _md5(configured / "models.yaml") == before
    assert "Defined twice" in r.text and "judge" in r.text


def test_detect_drafts_a_repo_without_writing_one(configured: Path, tmp_path: Path) -> None:
    """`eval-harness init` on the page. Every value is a guess, so it is a draft, not a save."""
    repos = configured / "repos.yaml"
    before = _md5(repos)
    client = TestClient(build_app())

    # A path that is not a checkout is refused in the page, not as a traceback.
    r = client.post("/setup/repos/detect", data={"path": str(tmp_path / "nope")})
    assert r.status_code == 422
    assert "not a directory" in r.text
    assert _md5(repos) == before

    repo = tmp_path / "some-app"
    (repo / ".git").mkdir(parents=True)
    (repo / "package.json").write_text('{"name": "some-app"}')
    (repo / "package-lock.json").write_text("{}")
    r = client.post("/setup/repos/detect", data={"path": str(repo), "key": "some-app"})
    assert r.status_code == 200
    assert "some-app:" in r.text, "the draft is in the editor"
    assert "CHECK" in r.text, "with its guesses marked"
    assert _md5(repos) == before, "and nothing on disk moved"


@pytest.mark.parametrize(
    "host,allowed",
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("127.0.0.53", True),
        ("0.0.0.0", False),
        ("192.168.1.10", False),
        ("::", False),
        ("eval-box.internal", False),
        ("", False),
    ],
)
def test_which_bind_addresses_may_write(host: str, allowed: bool) -> None:
    assert config_io.is_loopback(host) is allowed


def test_a_refused_save_hands_back_what_was_typed(configured: Path) -> None:
    """Reloading the form from disk after a 422 makes the reader retype the seven fields they
    got right to fix the one they got wrong, which is how the second mistake gets made."""
    client = TestClient(build_app())
    r = client.post(
        "/setup/models/demo-api",
        data={**A_MODEL, "model": "kept-through-the-error", "price_input": "2"},
    )
    assert r.status_code == 422
    assert "all four or none" in r.text
    assert "kept-through-the-error" in r.text, "the fields that were right are still there"
    assert 'name="price_input"' not in r.text, "pricing is not exposed in setup"
    # And only the form it came from is open, not every model's editor at once.
    assert r.text.count("<details open") == 1

    # Same for the repository form.
    r = client.post(
        "/setup/repos/demo-app",
        data={
            "github": "acme/demo-app",
            "local_path": "/tmp/typed-here",
            "linear_team": "DEMO",
            "image": "abeval/demo-app:node20",
            "dockerfile": "docker/node20.Dockerfile",
            "cpus": "4",
            "memory": "6g",
            "pids": "4096",
            "agent_authors": "demo-bot: nonsense",
        },
    )
    assert r.status_code == 422
    assert "is not a kind" in r.text
    assert "/tmp/typed-here" in r.text


def test_what_is_handed_back_never_includes_a_key(configured: Path) -> None:
    """The echo above must not become the way a credential gets onto the page.

    It did, the first time: handing the fields back handed back the key that was refused,
    into an input, the browser's history and any screenshot of it.
    """
    client = TestClient(build_app())
    key = "sk-ant-api03-CCCCCCCCCCCCCCCCCCCCCCCC"

    r = client.post("/setup/models/demo-api", data={**A_MODEL, "api_key_env": key})
    assert r.status_code == 422
    assert key not in r.text
    assert "demo-model-2" in r.text, "everything that is not the key still comes back"

    # The YAML editor is the worse version of the same thing: it would echo the whole file.
    r = client.post(
        "/setup/config/models.yaml/raw",
        data={"body": f"demo-api:\n  provider: anthropic\n  model: x\n  api_key_env: {key}\n"},
    )
    assert r.status_code == 422
    assert key not in r.text


def test_a_config_file_that_is_present_but_broken_is_not_offered_a_form(
    configured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A form edits one entry and saving validates the whole file, so a form over a file that
    does not load is a button that can only 422. Point at the text editor instead."""
    from eval_harness import doctor

    (configured / "models.yaml").write_text("demo-api:\n  provider: [unclosed\n")
    monkeypatch.setattr(
        doctor,
        "run_all",
        lambda: [
            doctor.Check(
                "models.yaml",
                "fail",
                f"{configured}/models.yaml: while parsing a flow node",
                "fix the YAML",
                kind="config",
            )
        ],
    )
    body = TestClient(build_app()).get("/setup?step=ready").text
    assert 'action="/setup/models"' not in body, (
        "no add-a-model form over a file that will not load"
    )
    assert "one bad entry stops the whole file loading" in body
    assert 'href="#models"' in body, "and a way to the editor that can fix it"


def test_the_fixture_config_was_never_written_to() -> None:
    """The guard that makes every test above safe to have written.

    `conftest` points the data root at `tests/fixtures`, so a write test that forgets to
    relocate edits the config every other test in the suite reads — it would pass, and take
    something else down with it later.
    """
    now = {
        p.name: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted((FIXTURES / "config").glob("*.yaml"))
    }
    assert now == FIXTURE_DIGESTS
    assert not list((FIXTURES / "config").glob("*.bak")), "nor backed up beside it"


def test_what_a_form_save_writes_is_what_a_person_would_have_written(configured: Path) -> None:
    """These files are read in diffs and edited by hand, so what this writes has to look like
    what a person would have written.

    Two things it got wrong first: PyYAML puts `- item` in the same column as the key above
    it, and a form posts every number as a float, so saving an unrelated field rewrote
    `cpus: 4` as `cpus: 4.0` — a line in a version-controlled diff that says nothing happened.
    """
    client = TestClient(build_app())
    r = client.post(
        "/setup/repos/demo-app",
        data={
            "github": "acme/demo-app",
            "local_path": "/tmp/moved",
            "linear_team": "DEMO",
            "image": "abeval/demo-app:node20",
            "dockerfile": "docker/node20.Dockerfile",
            "cpus": "4",
            "memory": "6g",
            "pids": "4096",
            "agent_authors": "demo-bot: agent",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text

    written = (configured / "repos.yaml").read_text()
    assert "    cpus: 4\n" in written, "an integral count stays an integer"
    assert "cpus: 4.0" not in written
    assert "  dep_dirs:\n    - dir: web\n" in written, "list items indent under their key"
    assert "\n  - dir: web\n" not in written
    # And it is still a file the harness loads, which is the only thing that must never change.
    assert load_repos(configured / "repos.yaml")["demo-app"].limits.cpus == 4


# ------------------------------------------------- a field the provider cannot send


def _with_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> TestClient:
    """A data root holding exactly this models.yaml, and a client pointed at it."""
    cfg = tmp_path / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "models.yaml").write_text(body)
    (cfg / "repos.yaml").write_text((FIXTURES / "config" / "repos.yaml").read_text())
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    return TestClient(build_app())


def test_the_form_offers_a_field_only_where_the_provider_can_send_it() -> None:
    """The gate the page renders from, and the same table the handler drops fields by."""
    assert "temperature" in config_io.fields_for("openai-compat")
    assert "reasoning_param" in config_io.fields_for("openai-compat")
    for provider in ("anthropic", "claude-code", "codex-cli"):
        assert "temperature" not in config_io.fields_for(provider), provider
        assert "reasoning_param" not in config_io.fields_for(provider), provider
    # claude-code reaches the ceiling through the CLI's environment; codex has no equivalent.
    assert "max_output_tokens" in config_io.fields_for("claude-code")
    assert "max_output_tokens" not in config_io.fields_for("codex-cli")
    # A provider with no adapter yet keeps every box rather than being quietly stripped.
    assert config_io.fields_for("something-new") == config_io.GATED_FIELDS


def test_the_page_hides_the_boxes_the_chosen_provider_cannot_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rendered server-side as well as gated by script, so it holds with JavaScript off."""
    client = _with_models(
        tmp_path,
        monkeypatch,
        "compat-one:\n"
        "  provider: openai-compat\n"
        "  model: m\n"
        "  base_url: http://x/v1\n"
        "  api_key_env: DEMO_API_KEY\n"
        "cli-one:\n"
        "  provider: codex-cli\n"
        "  model: m\n",
    )
    page = client.get("/setup").text
    compat = _form_for(page, "compat-one")
    cli = _form_for(page, "cli-one")
    assert 'data-for="temperature reasoning_param"' in compat
    assert "hidden" not in _gated_row(compat, "temperature reasoning_param")
    assert "hidden" in _gated_row(cli, "temperature reasoning_param")
    assert "hidden" in _gated_row(cli, "max_output_tokens")
    assert "hidden" not in _gated_row(compat, "max_output_tokens")


def _form_for(page: str, key: str) -> str:
    start = page.index(f'action="/setup/models/{key}"')
    return page[start : page.index("</form>", start)]


def _gated_row(form: str, data_for: str) -> str:
    """The opening tag of one gated group, where the `hidden` attribute would be."""
    start = form.index(f'data-for="{data_for}"')
    return form[start : form.index(">", start)]


def test_a_field_the_provider_cannot_send_is_not_written_even_if_posted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With JavaScript off every box posts; the handler is what keeps the file honest."""
    import yaml

    client = _with_models(tmp_path, monkeypatch, "judge:\n  provider: anthropic\n  model: m\n")
    reply = client.post(
        "/setup/models",
        data={
            "key": "cc",
            "provider": "claude-code",
            "model": "m",
            "effort": "high",
            "max_output_tokens": "16000",
            "temperature": "0.2",
            "reasoning_param": "reasoning_effort",
        },
        follow_redirects=False,
    )
    assert reply.status_code == 303, reply.text
    written = yaml.safe_load((tmp_path / "config" / "models.yaml").read_text())
    assert "temperature" not in written["cc"]
    assert "reasoning_param" not in written["cc"]
    assert written["cc"]["max_output_tokens"] == 16000
    # Which is the same thing the loader would have refused, reached without the refusal.
    assert load_models(tmp_path / "config" / "models.yaml")["cc"].temperature is None


def test_saving_a_codex_entry_keeps_the_field_the_form_could_not_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not offering a field is the point; deleting the line someone typed is not."""
    import yaml

    client = _with_models(
        tmp_path,
        monkeypatch,
        "terra:\n  provider: codex-cli\n  model: gpt\n  effort: high\n"
        "  max_output_tokens: 16000\n"
        "judge:\n  provider: anthropic\n  model: m\n",
    )
    reply = client.post(
        "/setup/models/terra",
        data={"key": "terra", "provider": "codex-cli", "model": "gpt", "effort": "medium"},
        follow_redirects=False,
    )
    assert reply.status_code == 303, reply.text
    written = yaml.safe_load((tmp_path / "config" / "models.yaml").read_text())
    assert written["terra"]["effort"] == "medium"
    assert written["terra"]["max_output_tokens"] == 16000


def test_the_openai_compat_fields_round_trip_through_the_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    client = _with_models(tmp_path, monkeypatch, "judge:\n  provider: anthropic\n  model: m\n")
    reply = client.post(
        "/setup/models",
        data={
            "key": "local",
            "provider": "openai-compat",
            "model": "qwen",
            "effort": "high",
            "max_output_tokens": "16000",
            "base_url": "http://gpu-box:11434/v1",
            "api_key_env": "OLLAMA_API_KEY",
            "temperature": "0.2",
            "reasoning_param": "reasoning.effort",
        },
        follow_redirects=False,
    )
    assert reply.status_code == 303, reply.text
    written = yaml.safe_load((tmp_path / "config" / "models.yaml").read_text())
    assert written["local"]["temperature"] == 0.2
    assert written["local"]["reasoning_param"] == "reasoning.effort"
    cfg = load_models(tmp_path / "config" / "models.yaml")["local"]
    assert cfg.temperature == 0.2 and cfg.reasoning_param == "reasoning.effort"


def test_the_default_reasoning_param_is_left_out_of_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`omit` is the default; writing it on every entry would be noise in a committed file."""
    import yaml

    client = _with_models(tmp_path, monkeypatch, "judge:\n  provider: anthropic\n  model: m\n")
    client.post(
        "/setup/models",
        data={
            "key": "local",
            "provider": "openai-compat",
            "model": "qwen",
            "effort": "high",
            "max_output_tokens": "16000",
            "base_url": "http://x/v1",
            "api_key_env": "OLLAMA_API_KEY",
            "reasoning_param": "omit",
            "temperature": "",
        },
    )
    written = yaml.safe_load((tmp_path / "config" / "models.yaml").read_text())
    assert "reasoning_param" not in written["local"]
    assert "temperature" not in written["local"]


def test_fresh_model_setup_is_blank_and_has_no_prices(fresh: Path) -> None:
    import re

    page = TestClient(build_app()).get("/setup?step=model")
    for name in ("key", "model", "effort", "max_output_tokens"):
        assert re.search(rf'name="{name}" value=""', page.text)
    assert '<option value="" selected>Choose a provider</option>' in page.text
    assert "Price per million tokens" not in page.text


def test_edit_without_price_fields_preserves_existing_prices(configured: Path) -> None:
    before = load_models()["demo-api"].price_per_mtok
    response = TestClient(build_app()).post("/setup/models/demo-api", data=A_MODEL)
    assert response.status_code in (200, 303)
    assert load_models()["demo-api"].price_per_mtok == before
