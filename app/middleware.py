import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("linear_rag-rag.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with timing, status, and key metadata."""

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        path = request.url.path
        method = request.method

        response = await call_next(request)

        elapsed_ms = (time.monotonic() - start) * 1000
        status = response.status_code

        log_data = {
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round(elapsed_ms, 1),
        }

        # Add query-specific metadata
        if path == "/retrieve" and hasattr(request, "_json"):
            try:
                body = await request.json()
                log_data["mode"] = body.get("mode", "unknown")
                log_data["top_k"] = body.get("top_k", 0)
            except Exception:
                pass

        if status >= 500:
            logger.error("request_failed", extra=log_data)
        elif status >= 400:
            logger.warning("request_error", extra=log_data)
        elif elapsed_ms > 5000:
            logger.warning("request_slow", extra=log_data)
        else:
            logger.info("request_ok", extra=log_data)

        return response
