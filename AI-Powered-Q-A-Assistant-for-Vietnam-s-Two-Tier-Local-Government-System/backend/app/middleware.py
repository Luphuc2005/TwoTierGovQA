"""
Custom middleware — request logging, rate limiting.
"""

import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.config import get_settings

settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every incoming request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)"
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per IP address."""

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for non-API routes
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < self.window
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )

        self.requests[client_ip].append(now)
        return await call_next(request)
