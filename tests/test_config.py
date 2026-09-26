from pathlib import Path

import pytest

from eval_harness.config import load_models, load_repos, local_path_env_var


def test_repos_yaml_loads_ab_test_app() -> None:
    cfg = load_repos()["demo-app"]
    assert cfg.github == "acme/demo-app"
    assert cfg.linear_team == "DEMO"
    assert {d.dir for d in cfg.dep_dirs} == {"web", "web/frontend", "web/script"}
    assert set(cfg.runners) == {"unit", "integration", "frontend", "script"}
    assert "~" not in str(cfg.local_path)


def test_runner_for_picks_unit_for_services_test() -> None:
    cfg = load_repos()["demo-app"]
    name, runner = cfg.runner_for(["web/services/demo/__tests__/ExampleService.test.ts"])
    assert name == "unit"
    assert runner.cwd == "web"
    assert runner.config == "vitest.unit.config.ts"


def test_runner_for_routes_frontend_script_and_integration() -> None:
    cfg = load_repos()["demo-app"]
    assert cfg.runner_for(["web/frontend/utils/__tests__/x.test.ts"])[0] == "frontend"
    assert cfg.runner_for(["web/test/integration/session/a.test.js"])[0] == "integration"
    assert cfg.runner_for(["web/script/module/exposure.test.ts"])[0] == "script"
    assert cfg.runner_for(["web/jobs/snapshot-writer/snapshot-writer.test.js"])[0] == "unit"
    assert cfg.runner_for(["web/test/helpers/srm-checker.test.js"])[0] == "unit"


def test_runner_for_rejects_mixed_runners() -> None:
    cfg = load_repos()["demo-app"]
    with pytest.raises(ValueError):
        cfg.runner_for(["web/frontend/a.test.ts", "web/services/__tests__/b.test.ts"])


def test_file_classification() -> None:
    cfg = load_repos()["demo-app"]
    assert cfg.is_test_file("web/services/demo/__tests__/ExampleService.test.ts")
    assert cfg.is_test_file("web/test/integration/session/a.test.js")
    assert not cfg.is_test_file("web/services/demo/ExampleService.ts")
    assert cfg.is_code_file("web/services/demo/ExampleService.ts")
    assert cfg.is_code_file("web/script/ab-test-v2.js")
    assert not cfg.is_code_file("specs/bug/DEMO-2693/plan.md")
    assert not cfg.is_code_file("web/jobs/bigquery-etl/queries/fct-events.sql")
    assert not cfg.is_code_file("web/jobs/billing-sync/Dockerfile")


def test_models_yaml_carries_providers_prices_and_a_judge() -> None:
    models = load_models()
    api = models["demo-api"]
    assert api.provider == "anthropic"
    assert api.price_per_mtok is not None and api.price_per_mtok.input == 2.0
    compat = models["demo-compat"]
    assert compat.base_url == "https://example.invalid/v1"
    assert compat.api_key_env == "DEMO_API_KEY", "a key is named, never inlined"
    assert "judge" in models, "scoring needs one"


def test_the_shipped_example_models_file_is_valid() -> None:
    """It is what a new user copies, so a typo in it is a broken first run."""
    from eval_harness import PROJECT_ROOT

    example = PROJECT_ROOT / "config" / "models.example.yaml"
    models = load_models(example)
    assert "judge" in models
    assert all(m.api_key_env != "" for m in models.values() if m.provider == "openai-compat")


def test_runner_groups_splits_mixed_runners_in_config_order() -> None:
    cfg = load_repos()["demo-app"]
    groups = cfg.runner_groups(
        [
            "web/routes/public/__tests__/ga4-metadata.test.ts",
            "web/script/module/group-assignment.test.js",
            "web/test/integration/session/a.test.js",
        ]
    )
    assert [(n, fs) for n, _, fs in groups] == [
        ("integration", ["web/test/integration/session/a.test.js"]),
        ("script", ["web/script/module/group-assignment.test.js"]),
        ("unit", ["web/routes/public/__tests__/ga4-metadata.test.ts"]),
    ]


def test_local_path_env_var_overrides_the_clone_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert local_path_env_var("demo-app") == "ABEVAL_PATH_DEMO_APP"
    monkeypatch.setenv("ABEVAL_PATH_DEMO_APP", str(tmp_path))
    assert load_repos()["demo-app"].local_path == tmp_path


def test_data_root_defaults_to_the_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval_harness import PROJECT_ROOT, paths

    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    paths.reset_cache()
    assert paths.data_root() == PROJECT_ROOT
    assert paths.cases_dir() == PROJECT_ROOT / "data" / "cases"
    assert paths.results_root() == PROJECT_ROOT / "results"


def test_data_root_relocates_every_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One env var moves cases, datasets, curation, raw cache and results together."""
    from eval_harness import paths

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    for got in (
        paths.cases_dir(),
        paths.datasets_dir(),
        paths.curation_path(),
        paths.splits_path(),
        paths.reviews_dir(),
        paths.raw_cache("repo"),
        paths.results_root(),
        paths.jobs_dir(),
    ):
        assert tmp_path in got.parents or got.parent == tmp_path, got


def test_accessors_are_read_at_call_time_not_import_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A module constant would freeze the root before a caller could set it."""
    from eval_harness import paths
    from eval_harness.collect import cases

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    (tmp_path / "data" / "cases").mkdir(parents=True)
    assert cases.load_case.__defaults__ == (None,)
    assert paths.cases_dir() == tmp_path / "data" / "cases"
