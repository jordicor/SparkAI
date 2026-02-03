"""
Security Middleware for Scanner/Bot Protection.

Provides multi-layer protection:
1. Instant block for known malicious patterns (wp-*, .env, .git, etc.)
2. Progressive block based on 404 accumulation (10 404s in 3 min = ban)
3. Works with or without nginx/Cloudflare (detects IP from appropriate header)

Compatible with:
- FastAPI standalone (direct connection)
- FastAPI + nginx (X-Forwarded-For, X-Real-IP)
- FastAPI + nginx + Cloudflare (CF-Connecting-IP)
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Set, Dict, Tuple
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class SecurityConfig:
    """Security middleware configuration."""

    # Instant block patterns - 1 request = immediate ban
    # These are ALWAYS malicious, no legitimate user would request them
    INSTANT_BLOCK_PATTERNS = [
        # WordPress
        r"^/wp-",
        r"^/wordpress",
        r"/xmlrpc\.php",
        r"/wp-login\.php",
        r"/wp-config\.php",

        # PHP admin panels
        r"^/phpmyadmin",
        r"^/pma",
        r"^/myadmin",
        r"^/mysql",
        r"^/phpMyAdmin",

        # Hidden files and configs
        r"/\.git",
        r"/\.env",
        r"/\.aws",
        r"/\.ssh",
        r"/\.htaccess",
        r"/\.htpasswd",
        r"/\.DS_Store",
        r"/\.svn",
        r"/\.hg",

        # Config files
        r"/config\.(php|yml|yaml|json|ini|xml|bak)$",
        r"/settings\.(php|yml|yaml|json|ini|xml)$",
        r"/database\.(php|yml|yaml|json|ini|xml)$",
        r"/credentials",
        r"/secrets",

        # Shell/exploit attempts
        r"/(shell|cmd|exec|eval|system|passthru)\.php",
        r"/c99\.php",
        r"/r57\.php",
        r"/alfa\.php",
        r"/wso\.php",
        r"/b374k",

        # CGI attacks
        r"^/cgi-bin/",
        r"^/cgi/",

        # Admin panels
        r"^/admin\.php",
        r"^/administrator",
        r"^/admin/.*\.(php|asp|aspx|jsp)$",
        r"^/manager/html",
        r"^/manager/status",

        # Backup files
        r"\.(sql|bak|backup|old|orig|save|swp|tmp)$",
        r"\.(tar|tar\.gz|tgz|zip|rar|7z)$",
        r"~$",  # Backup files like config.php~

        # AWS/Cloud metadata
        r"/latest/meta-data",
        r"/169\.254\.169\.254",

        # Common CMS
        r"^/joomla",
        r"^/drupal",
        r"^/magento",
        r"^/typo3",

        # Java/Tomcat (specific endpoints, not generic /manager/ which conflicts with app routes)
        r"^/manager/html",
        r"^/manager/status",
        r"^/manager/text",
        r"^/manager/jmxproxy",
        r"^/jenkins",
        r"^/hudson",
        r"^/solr",
        r"^/actuator",
        r"^/console",

        # Debug endpoints
        r"/phpinfo\.php",
        r"/info\.php",
        r"/test\.php",
        r"/debug",
        r"/trace",

        # GraphQL introspection abuse
        r"/graphql.*introspection",

        # Kubernetes/Docker
        r"^/k8s/",
        r"^/kubernetes/",
        r"^/docker",
        r"^/portainer",
    ]

    # Compiled patterns for performance
    COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INSTANT_BLOCK_PATTERNS]

    # Whitelist - these 404s don't count toward threshold
    # Browsers/bots request these automatically
    WHITELIST_PATHS: Set[str] = {
        "/favicon.ico",
        "/favicon.png",
        "/apple-touch-icon.png",
        "/apple-touch-icon-precomposed.png",
        "/apple-touch-icon-120x120.png",
        "/apple-touch-icon-152x152.png",
        "/apple-touch-icon-180x180.png",
        "/robots.txt",
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/browserconfig.xml",
        "/manifest.json",
        "/site.webmanifest",
        "/humans.txt",
        "/ads.txt",
        "/security.txt",
    }

    WHITELIST_PREFIXES = [
        "/.well-known/",  # ACME, security.txt, etc.
    ]

    # Landing page routes - these are user-generated content, skip pattern blocking
    # Only 404 accumulation applies to these routes
    LANDING_PAGE_PATTERNS = [
        re.compile(r"^/p/[a-zA-Z0-9]{8}/"),      # Public landing pages: /p/{public_id}/{slug}/
        re.compile(r"^/\d+/[^/]+/"),             # Legacy prompt routes: /{prompt_id}/{prompt_name}/
    ]

    # Known legitimate bots (more permissive threshold)
    KNOWN_BOTS = [
        "Googlebot",
        "Bingbot",
        "Slurp",      # Yahoo
        "DuckDuckBot",
        "Yandex",
        "Baiduspider",
        "facebot",    # Facebook
        "Twitterbot",
        "LinkedInBot",
        "Applebot",
    ]

    # Thresholds: (max_404s, window_minutes)
    NORMAL_THRESHOLD: Tuple[int, int] = (10, 3)   # 10 404s in 3 min
    BOT_THRESHOLD: Tuple[int, int] = (50, 3)      # 50 404s in 3 min for known bots

    # Block duration
    BLOCK_DURATION_HOURS: int = 24

    # Response for blocked requests (444 = nginx silent drop, 403 for others)
    BLOCK_STATUS_CODE: int = 403
    BLOCK_MESSAGE: str = "Forbidden"


# =============================================================================
# IP Tracking Storage (in-memory, single process)
# =============================================================================

class SecurityTracker:
    """
    Tracks 404 counts and blocked IPs.

    For multi-process/distributed setups, replace with Redis.
    """

    def __init__(self):
        # {ip: [timestamp, timestamp, ...]}
        self._404_counts: Dict[str, list] = defaultdict(list)

        # {ip: unblock_timestamp}
        self._blocked_ips: Dict[str, datetime] = {}

        # Last cleanup time
        self._last_cleanup = datetime.now()

    def is_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        if ip not in self._blocked_ips:
            return False

        unblock_time = self._blocked_ips[ip]
        if datetime.now() >= unblock_time:
            # Block expired
            del self._blocked_ips[ip]
            return False

        return True

    def block_ip(self, ip: str, hours: int = 24, reason: str = ""):
        """Block an IP for specified duration."""
        unblock_time = datetime.now() + timedelta(hours=hours)
        self._blocked_ips[ip] = unblock_time
        logger.warning(f"SECURITY: Blocked IP {ip} for {hours}h. Reason: {reason}")

    def record_404(self, ip: str) -> int:
        """Record a 404 and return current count in window."""
        now = datetime.now()
        self._404_counts[ip].append(now)
        self._cleanup_old_entries()
        return len(self._404_counts[ip])

    def get_404_count(self, ip: str, window_minutes: int) -> int:
        """Get 404 count for IP within time window."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)

        # Filter to window
        timestamps = self._404_counts.get(ip, [])
        recent = [t for t in timestamps if t > cutoff]
        self._404_counts[ip] = recent

        return len(recent)

    def _cleanup_old_entries(self):
        """Periodic cleanup to prevent memory bloat."""
        now = datetime.now()

        # Only cleanup every 5 minutes
        if (now - self._last_cleanup).total_seconds() < 300:
            return

        # Clean 404 counts older than 1 hour
        cutoff = now - timedelta(hours=1)
        keys_to_delete = []

        for ip, timestamps in self._404_counts.items():
            self._404_counts[ip] = [t for t in timestamps if t > cutoff]
            if not self._404_counts[ip]:
                keys_to_delete.append(ip)

        for key in keys_to_delete:
            del self._404_counts[key]

        # Clean expired blocks
        expired_blocks = [
            ip for ip, unblock_time in self._blocked_ips.items()
            if now >= unblock_time
        ]
        for ip in expired_blocks:
            del self._blocked_ips[ip]

        self._last_cleanup = now

        if keys_to_delete or expired_blocks:
            logger.debug(
                f"Security cleanup: removed {len(keys_to_delete)} 404 trackers, "
                f"{len(expired_blocks)} expired blocks"
            )

    def get_stats(self) -> dict:
        """Get current security stats (for admin/monitoring)."""
        return {
            "blocked_ips_count": len(self._blocked_ips),
            "tracked_ips_count": len(self._404_counts),
            "blocked_ips": {
                ip: unblock.isoformat()
                for ip, unblock in self._blocked_ips.items()
            }
        }


# Singleton tracker instance
_tracker = SecurityTracker()


# =============================================================================
# Helper Functions
# =============================================================================

def get_client_ip(request: Request) -> str:
    """
    Extract real client IP, considering reverse proxies.

    Priority:
    1. CF-Connecting-IP (Cloudflare)
    2. X-Forwarded-For (nginx/proxies - first IP in chain)
    3. X-Real-IP (nginx)
    4. Direct connection
    """
    # Cloudflare
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # X-Forwarded-For (can have multiple: client, proxy1, proxy2)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # X-Real-IP from nginx
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Direct connection
    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def is_known_bot(request: Request) -> bool:
    """Check if request is from a known legitimate bot."""
    user_agent = request.headers.get("User-Agent", "")
    return any(bot.lower() in user_agent.lower() for bot in SecurityConfig.KNOWN_BOTS)


def is_whitelisted_path(path: str) -> bool:
    """Check if path is whitelisted (shouldn't count as 404)."""
    path_lower = path.lower()

    # Exact match
    if path_lower in SecurityConfig.WHITELIST_PATHS:
        return True

    # Prefix match
    for prefix in SecurityConfig.WHITELIST_PREFIXES:
        if path_lower.startswith(prefix):
            return True

    return False


def matches_instant_block_pattern(path: str) -> Optional[str]:
    """
    Check if path matches any instant-block pattern.
    Returns the matched pattern string or None.
    """
    for i, pattern in enumerate(SecurityConfig.COMPILED_PATTERNS):
        if pattern.search(path):
            return SecurityConfig.INSTANT_BLOCK_PATTERNS[i]
    return None


def is_landing_page_route(path: str) -> bool:
    """
    Check if path is a landing page route.

    Landing pages are user-generated content where the slug comes from
    the prompt name. We skip instant-block pattern matching for these
    to avoid false positives (e.g., a prompt named "docker-tutorial"
    would match the /docker pattern).

    404 accumulation still applies to catch actual scanners.
    """
    for pattern in SecurityConfig.LANDING_PAGE_PATTERNS:
        if pattern.match(path):
            return True
    return False


# =============================================================================
# Main Middleware
# =============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for scanner/bot protection.

    Must be added EARLY in the middleware chain to intercept
    malicious requests before they consume resources.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        ip = get_client_ip(request)
        path = request.url.path

        # 1. Check if IP is already blocked
        if _tracker.is_blocked(ip):
            logger.debug(f"SECURITY: Blocked request from banned IP {ip}: {path}")
            return Response(
                content=SecurityConfig.BLOCK_MESSAGE,
                status_code=SecurityConfig.BLOCK_STATUS_CODE
            )

        # 2. Check for instant-block patterns
        # Skip pattern matching for landing page routes (user-generated slugs)
        # to avoid false positives like /p/xxx/docker-tutorial/
        if not is_landing_page_route(path):
            matched_pattern = matches_instant_block_pattern(path)
            if matched_pattern:
                _tracker.block_ip(
                    ip,
                    hours=SecurityConfig.BLOCK_DURATION_HOURS,
                    reason=f"Instant block pattern: {matched_pattern} (path: {path})"
                )
                return Response(
                    content=SecurityConfig.BLOCK_MESSAGE,
                    status_code=SecurityConfig.BLOCK_STATUS_CODE
                )

        # 3. Process the request normally
        response = await call_next(request)

        # 4. After response: check if 404 and track
        if response.status_code == 404:
            # Skip whitelisted paths
            if not is_whitelisted_path(path):
                # Get appropriate threshold
                if is_known_bot(request):
                    max_404s, window_min = SecurityConfig.BOT_THRESHOLD
                else:
                    max_404s, window_min = SecurityConfig.NORMAL_THRESHOLD

                # Record and check
                _tracker.record_404(ip)
                count = _tracker.get_404_count(ip, window_min)

                logger.debug(f"SECURITY: 404 from {ip}: {path} (count: {count}/{max_404s})")

                if count >= max_404s:
                    _tracker.block_ip(
                        ip,
                        hours=SecurityConfig.BLOCK_DURATION_HOURS,
                        reason=f"Too many 404s: {count} in {window_min} min (last: {path})"
                    )
                    # Note: This response already went out as 404
                    # The NEXT request from this IP will be blocked

        return response


# =============================================================================
# API for external access (admin endpoints, etc.)
# =============================================================================

def get_security_stats() -> dict:
    """Get current security statistics."""
    return _tracker.get_stats()


def manually_block_ip(ip: str, hours: int = 24, reason: str = "Manual block"):
    """Manually block an IP (for admin use)."""
    _tracker.block_ip(ip, hours, reason)


def manually_unblock_ip(ip: str) -> bool:
    """Manually unblock an IP. Returns True if IP was blocked."""
    if ip in _tracker._blocked_ips:
        del _tracker._blocked_ips[ip]
        logger.info(f"SECURITY: Manually unblocked IP {ip}")
        return True
    return False


def is_ip_blocked(ip: str) -> bool:
    """Check if an IP is currently blocked."""
    return _tracker.is_blocked(ip)
