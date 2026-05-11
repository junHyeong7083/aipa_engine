"""
Request ID Tracking & Rate Limiting Middleware

Pure ASGI middleware (BaseHTTPMiddleware 사용하지 않음 - Starlette hanging 버그 방지)
"""

import logging
import time
import uuid
from collections import defaultdict

from starlette.types import ASGIApp, Receive, Scope, Send

from .config import get_settings
from .logging_config import request_id_var

logger = logging.getLogger("aipa.middleware")


def get_request_id() -> str:
    return request_id_var.get() or ""


# ---------------------------------------------------------------------------
# Request ID Middleware (Pure ASGI)
# ---------------------------------------------------------------------------

class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract or generate request ID
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())
        token = request_id_var.set(request_id)

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)


# ---------------------------------------------------------------------------
# Rate Limiting (Pure ASGI)
# ---------------------------------------------------------------------------

class _ClientBucket:
    __slots__ = ("timestamps",)

    def __init__(self):
        self.timestamps: list[float] = []

    def allow(self, now: float, window: float, max_requests: int) -> bool:
        cutoff = now - window
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        if len(self.timestamps) >= max_requests:
            return False
        self.timestamps.append(now)
        return True

    def retry_after(self, now: float, window: float) -> int:
        if not self.timestamps:
            return 0
        oldest_in_window = self.timestamps[0]
        return max(1, int(oldest_in_window + window - now) + 1)


class RateLimitMiddleware:
    CLEANUP_INTERVAL = 300

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._buckets: dict[tuple[str, str], _ClientBucket] = defaultdict(_ClientBucket)
        self._last_cleanup: float = time.time()

    def _get_client_ip(self, scope: Scope) -> str:
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for", b"").decode()
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _cleanup_if_needed(self, now: float) -> None:
        if now - self._last_cleanup < self.CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        stale_keys = [
            k for k, bucket in self._buckets.items()
            if not bucket.timestamps or (now - bucket.timestamps[-1]) > 120
        ]
        for k in stale_keys:
            del self._buckets[k]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        now = time.time()
        self._cleanup_if_needed(now)

        client_ip = self._get_client_ip(scope)
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        window = 60.0

        if method == "POST" and path.rstrip("/") == f"{settings.api_prefix}/simulations":
            key = (client_ip, "simulation")
            max_requests = settings.simulation_rate_limit_per_minute
        else:
            key = (client_ip, "general")
            max_requests = settings.rate_limit_per_minute

        bucket = self._buckets[key]

        if not bucket.allow(now, window, max_requests):
            retry_after = bucket.retry_after(now, window)
            body = b'{"detail":"Too Many Requests"}'
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", str(retry_after).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
