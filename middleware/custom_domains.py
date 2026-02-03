"""
Custom Domain Middleware for Landing Pages.

Handles routing for custom domains pointing to prompt landing pages.
Uses in-memory cache with TTL to minimize database lookups.
"""

import os
from typing import Optional, Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from cachetools import TTLCache

# Cache: domain -> prompt_data (5 minute TTL, max 1000 entries)
_domain_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

# Primary domains that should skip custom domain lookup
_primary_domains: set = set()


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
