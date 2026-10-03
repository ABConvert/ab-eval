"""Happy-path dashboard requests carry the Origin a browser sends with its forms."""

from starlette.testclient import TestClient as StarletteTestClient
from starlette.types import ASGIApp


class TestClient(StarletteTestClient):
    __test__ = False

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(
            app, base_url="http://127.0.0.1:8765", headers={"origin": "http://127.0.0.1:8765"}
        )
