"""
Vercel Serverless Function Entrypoint (api/index.py)
Exposes the FastAPI application from backend/main.py for Vercel Python Runtime.
"""

import os
import sys

# Add backend directory to Python path so all backend modules resolve cleanly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import app


class StripApiPrefixMiddleware:
    """ASGI middleware that strips leading '/api' prefix for Vercel rewrites."""

    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            path = scope.get("path", "")
            if path == "/api":
                scope = dict(scope, path="/")
            elif path.startswith("/api/"):
                scope = dict(scope, path=path[4:])
        await self.app(scope, receive, send)


app.add_middleware(StripApiPrefixMiddleware)
