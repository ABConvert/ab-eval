"""Synthetic boundary probes: no provider requests, jobs, or private benchmark data."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from starlette.routing import Route
from starlette.testclient import TestClient

from eval_harness import paths
from eval_harness.dashboard import jobs
from eval_harness.dashboard.app import build_app


@pytest.fixture
def private_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.reset_cache()
    monkeypatch.setattr(jobs, "queue", Mock(side_effect=AssertionError("job queue reached")))
    return tmp_path


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.2", "dashboard.example"])
def test_dashboard_refuses_network_bind(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        build_app(host)


@pytest.mark.parametrize("host", ["attacker.invalid", "127.0.0.1.attacker.invalid"])
def test_dashboard_rejects_untrusted_host_before_reading_data(
    host: str, private_root: Path
) -> None:
    client = TestClient(build_app(), base_url=f"http://{host}:8765")
    assert client.get("/setup").status_code == 400
    assert (
        client.post(
            "/setup/config/models.yaml/raw",
            headers={"origin": f"http://{host}:8765", "sec-fetch-site": "same-origin"},
            data={"body": "{}"},
        ).status_code
        == 400
    )
    assert list(private_root.iterdir()) == []


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://attacker.invalid", "sec-fetch-site": "cross-site"},
        {"origin": "https://attacker.invalid"},  # clients without Fetch Metadata
        {"origin": "null"},
        {"origin": "http://127.0.0.1:9999"},  # a different local application
        {"origin": "http://127.0.0.1:8765", "sec-fetch-site": "same-site"},
        {},  # missing provenance is not approval
    ],
)
def test_every_write_route_checks_origin_before_side_effects(
    headers: dict[str, str], private_root: Path
) -> None:
    app = build_app()
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    for route in app.routes:
        if not isinstance(route, Route) or "POST" not in (route.methods or set()):
            continue
        import re

        path = re.sub(r"\{[^}]+\}", "synthetic", route.path)
        response = client.post(path, headers=headers, data={"name": "synthetic"})
        assert response.status_code == 403, path
    assert list(private_root.iterdir()) == []


@pytest.mark.parametrize(
    "base", ["http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"]
)
def test_same_origin_form_still_writes(base: str, private_root: Path) -> None:
    client = TestClient(
        build_app(),
        base_url="http://127.0.0.1:8765",
        headers={"host": base.removeprefix("http://")},
    )
    response = client.post(
        "/datasets/create",
        headers={"origin": base},
        data={"name": "synthetic"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert (paths.datasets_dir() / "synthetic.json").exists()
