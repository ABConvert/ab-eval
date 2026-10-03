"""Local dashboard boundary, applied before any route can read data or start work."""

from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from eval_harness.dashboard.config_io import is_loopback


def _origin(value: str) -> tuple[str, str, int] | None:
    try:
        url = urlsplit(value)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.path
            or url.query
            or url.fragment
        ):
            return None
        return url.scheme, url.hostname, url.port or (443 if url.scheme == "https" else 80)
    except ValueError:
        return None


class DashboardBoundary:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        hosts = headers.getlist("host")
        target = _origin(f"{scope['scheme']}://{hosts[0]}") if len(hosts) == 1 else None
        # A loopback bind alone does not stop DNS rebinding: the Host must be local too.
        if target is None or not is_loopback(target[1]):
            await PlainTextResponse("Invalid dashboard host", status_code=400)(scope, receive, send)
            return
        if scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            origins = headers.getlist("origin")
            origin = _origin(origins[0]) if len(origins) == 1 else None
            site = headers.get("sec-fetch-site")
            if origin != target or site not in (None, "same-origin", "none"):
                await PlainTextResponse(
                    "That request arrived from another site or has no matching Origin. "
                    "Reload the dashboard and try again. Nothing was changed.",
                    status_code=403,
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)
