"""
Custom Domain Middleware for Landing Pages.

Handles routing for custom domains pointing to prompt landing pages.
Uses in-memory cache with TTL to minimize database lookups.

Static files for custom domains are served directly from this middleware
to bypass FastAPI's StaticFiles mount at /static which would otherwise
intercept and look in the global data/static/ directory.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, Response
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Media type mapping for static file serving
_MEDIA_TYPES = {
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.webp': 'image/webp',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
    '.mp3': 'audio/mpeg',
    '.mp4': 'video/mp4',
    '.json': 'application/json',
}

# Cache: domain -> prompt_data (5 minute TTL, max 1000 entries)
_domain_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

# Primary domains that should skip custom domain lookup
_primary_domains: set = set()


def _build_prompt_path(username: str, prompt_id: int, prompt_name: str) -> Path:
    """Build filesystem path to a prompt's landing page directory."""
    from common import generate_user_hash, sanitize_name, DATA_DIR

    hash_prefix1, hash_prefix2, user_hash = generate_user_hash(username)
    padded_id = f"{prompt_id:07d}"
    safe_name = sanitize_name(prompt_name)

    return (
        DATA_DIR / "users" / hash_prefix1 / hash_prefix2 / user_hash
        / "prompts" / padded_id[:3] / f"{padded_id[3:]}_{safe_name}"
    )


def set_primary_domains(domains: list):
    """
    Set the primary domains (call during app startup).
    Requests to these domains skip the custom domain DB lookup.
    """
    global _primary_domains
    _primary_domains = {d.lower().strip() for d in domains if d}


def is_primary_domain(host: str) -> bool:
    """Check if host is a primary domain."""
    host = host.lower().strip()
    return host in _primary_domains or host in ("localhost", "127.0.0.1", "")


class CustomDomainMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle custom domain routing for prompt landing pages.

    Flow:
    1. Extract Host header
    2. If Host is primary domain, pass through (no DB lookup)
    3. If Host is different, check cache -> DB for custom domain mapping
    4. If verified and active, inject prompt data into request.state
    5. Otherwise, return 404
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        host = request.headers.get("host", "").lower().split(":")[0]

        # Skip for primary domains
        if is_primary_domain(host):
            return await call_next(request)

        # Check if this is a custom domain
        domain_data = await self._get_domain_data(host)

        if domain_data is None:
            # Not a known custom domain -> 404
            return Response(
                content=self._get_404_html(),
                status_code=404,
                media_type="text/html"
            )

        # Inject prompt data into request state
        request.state.custom_domain = True
        request.state.custom_domain_host = host
        request.state.prompt_id = domain_data["prompt_id"]
        request.state.prompt_name = domain_data["prompt_name"]
        request.state.username = domain_data["username"]
        request.state.public_id = domain_data["public_id"]

        # Serve static files directly to bypass the global StaticFiles mount.
        # Without this, /static/* requests hit app.mount("/static", StaticFiles(...))
        # which looks in data/static/ (global) instead of the prompt's directory.
        path = request.url.path
        if path.startswith("/static/"):
            return self._serve_landing_static(domain_data, path[8:])  # strip "/static/"

        return await call_next(request)

    async def _get_domain_data(self, domain: str) -> Optional[Dict]:
        """Get prompt data for a custom domain (cached)."""
        # Check cache first
        if domain in _domain_cache:
            return _domain_cache[domain]

        # DB lookup
        from database import get_db_connection

        async with get_db_connection(readonly=True) as conn:
            cursor = await conn.execute("""
                SELECT
                    pcd.prompt_id,
                    p.name as prompt_name,
                    p.public_id,
                    u.username
                FROM PROMPT_CUSTOM_DOMAINS pcd
                JOIN PROMPTS p ON pcd.prompt_id = p.id
                JOIN USERS u ON p.created_by_user_id = u.id
                WHERE pcd.custom_domain = ?
                  AND pcd.is_active = 1
                  AND pcd.verification_status = 1
            """, (domain,))
            result = await cursor.fetchone()

        if result:
            data = {
                "prompt_id": result[0],
                "prompt_name": result[1],
                "public_id": result[2],
                "username": result[3]
            }
            _domain_cache[domain] = data
            return data

        return None

    def _serve_landing_static(self, domain_data: Dict, resource_path: str) -> Response:
        """
        Serve a static file from the prompt's landing page directory.
        Returns FileResponse on success, 404 on failure.
        """
        if not resource_path or ".." in resource_path:
            return Response(status_code=404)

        prompt_dir = _build_prompt_path(
            domain_data["username"],
            domain_data["prompt_id"],
            domain_data["prompt_name"],
        )
        static_path = prompt_dir / "static" / resource_path

        # Security first: validate path stays within prompt directory
        try:
            resolved = static_path.resolve(strict=False)
            resolved.relative_to(prompt_dir.resolve())
        except (ValueError, OSError):
            return Response(status_code=404)

        if not resolved.is_file():
            return Response(status_code=404)

        suffix = resolved.suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, 'application/octet-stream')

        return FileResponse(
            resolved,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    def _get_404_html(self) -> str:
        """Return 404 HTML for unknown domains."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Domain Not Found</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            background: #f5f5f5;
        }
        .container { text-align: center; }
        h1 { font-size: 6rem; color: #ddd; margin: 0; font-weight: 200; }
        p { color: #888; font-size: 1.2rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>404</h1>
        <p>Domain not configured</p>
    </div>
</body>
</html>"""


def invalidate_domain_cache(domain: str):
    """Invalidate cache for a specific domain (call after updates)."""
    domain = domain.lower().strip()
    if domain in _domain_cache:
        del _domain_cache[domain]


def clear_domain_cache():
    """Clear entire domain cache (for maintenance)."""
    _domain_cache.clear()
