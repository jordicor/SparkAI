"""
Unified CAPTCHA verification service.
Supports: Cloudflare Turnstile, Google reCAPTCHA v3

Configuration via .env:
    CAPTCHA_PROVIDER: turnstile | recaptcha | none
    TURNSTILE_SITE_KEY: Cloudflare Turnstile site key
    TURNSTILE_SECRET_KEY: Cloudflare Turnstile secret key
    RECAPTCHA_SITE_KEY: Google reCAPTCHA v3 site key
    RECAPTCHA_SECRET_KEY: Google reCAPTCHA v3 secret key
"""
import os
import httpx
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Runtime toggle - can be changed without restart (admin only)
_captcha_runtime_enabled = True

# Configuration
CAPTCHA_PROVIDER = os.getenv("CAPTCHA_PROVIDER", "none").lower().strip()
TURNSTILE_SITE_KEY = os.getenv("TURNSTILE_SITE_KEY", "")
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "")
RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY", "")
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY", "")

# Verification URLs
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"

# reCAPTCHA v3 score threshold (0.0 - 1.0, lower = more suspicious)
RECAPTCHA_SCORE_THRESHOLD = float(os.getenv("RECAPTCHA_SCORE_THRESHOLD", "0.3"))


async def verify_captcha(token: str, ip: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verify CAPTCHA token based on configured provider.

    Args:
        token: The CAPTCHA response token from the frontend
        ip: Optional client IP address for additional verification

    Returns:
        Tuple[bool, str]: (success, error_message)
            - success: True if verification passed or CAPTCHA is disabled
            - error_message: Empty string on success, error description on failure
    """
    # Check runtime toggle first
    if not _captcha_runtime_enabled:
        return True, ""

    if CAPTCHA_PROVIDER == "none" or not CAPTCHA_PROVIDER:
        return True, ""

    if not token or not token.strip():
        logger.warning("CAPTCHA verification attempted without token")
        return False, "CAPTCHA verification required"

    if CAPTCHA_PROVIDER == "turnstile":
        return await _verify_turnstile(token, ip)
    elif CAPTCHA_PROVIDER == "recaptcha":
        return await _verify_recaptcha(token, ip)
    else:
        logger.warning(f"Unknown CAPTCHA provider configured: {CAPTCHA_PROVIDER}")
        return True, ""  # Fail open if misconfigured


async def _verify_turnstile(token: str, ip: Optional[str]) -> Tuple[bool, str]:
    """
    Verify Cloudflare Turnstile token.

    Turnstile returns a simple success/failure response.
    """
    if not TURNSTILE_SECRET_KEY:
        logger.error("TURNSTILE_SECRET_KEY not configured - skipping verification")
        return True, ""  # Fail open if not configured

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "secret": TURNSTILE_SECRET_KEY,
                "response": token,
            }
            if ip:
                payload["remoteip"] = ip

            response = await client.post(TURNSTILE_VERIFY_URL, data=payload)
            result = response.json()

            if result.get("success"):
                logger.debug("Turnstile verification successful")
                return True, ""
            else:
                errors = result.get("error-codes", [])
                logger.warning(f"Turnstile verification failed: {errors}")
                return False, "CAPTCHA verification failed. Please try again."

    except httpx.TimeoutException:
        logger.error("Turnstile verification timeout")
        return True, ""  # Fail open on timeout
    except Exception as e:
        logger.error(f"Turnstile verification error: {e}")
        return True, ""  # Fail open on network errors


async def _verify_recaptcha(token: str, ip: Optional[str]) -> Tuple[bool, str]:
    """
    Verify Google reCAPTCHA v3 token.

    reCAPTCHA v3 returns a score between 0.0 (bot) and 1.0 (human).
    We use RECAPTCHA_SCORE_THRESHOLD to determine pass/fail.
    """
    if not RECAPTCHA_SECRET_KEY:
        logger.error("RECAPTCHA_SECRET_KEY not configured - skipping verification")
        return True, ""  # Fail open if not configured

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "secret": RECAPTCHA_SECRET_KEY,
                "response": token,
            }
            if ip:
                payload["remoteip"] = ip

            response = await client.post(RECAPTCHA_VERIFY_URL, data=payload)
            result = response.json()

            if not result.get("success"):
                errors = result.get("error-codes", [])
                logger.warning(f"reCAPTCHA verification failed: {errors}")
                return False, "CAPTCHA verification failed. Please try again."

            # reCAPTCHA v3 score check
            score = result.get("score", 0.0)
            action = result.get("action", "unknown")

            if score < RECAPTCHA_SCORE_THRESHOLD:
                logger.warning(f"reCAPTCHA low score: {score} for action '{action}'")
                return False, "CAPTCHA verification failed. Please try again."

            logger.debug(f"reCAPTCHA verification successful: score={score}, action={action}")
            return True, ""

    except httpx.TimeoutException:
        logger.error("reCAPTCHA verification timeout")
        return True, ""  # Fail open on timeout
    except Exception as e:
        logger.error(f"reCAPTCHA verification error: {e}")
        return True, ""  # Fail open on network errors


def get_captcha_config() -> dict:
    """
    Get CAPTCHA configuration for frontend templates.

    Returns:
        dict with:
            - provider: 'turnstile' | 'recaptcha' | 'none'
            - site_key: The public site key for the provider
            - enabled: Boolean indicating if CAPTCHA is active
    """
    # Check runtime toggle first
    if not _captcha_runtime_enabled:
        return {
            "provider": "none",
            "site_key": "",
            "enabled": False,
        }

    if CAPTCHA_PROVIDER == "turnstile" and TURNSTILE_SITE_KEY:
        return {
            "provider": "turnstile",
            "site_key": TURNSTILE_SITE_KEY,
            "enabled": True,
        }
    elif CAPTCHA_PROVIDER == "recaptcha" and RECAPTCHA_SITE_KEY:
        return {
            "provider": "recaptcha",
            "site_key": RECAPTCHA_SITE_KEY,
            "enabled": True,
        }
    else:
        return {
            "provider": "none",
            "site_key": "",
            "enabled": False,
        }


def is_captcha_enabled() -> bool:
    """Check if CAPTCHA verification is enabled (considers runtime toggle)."""
    if not _captcha_runtime_enabled:
        return False
    return CAPTCHA_PROVIDER in ("turnstile", "recaptcha") and bool(
        (CAPTCHA_PROVIDER == "turnstile" and TURNSTILE_SECRET_KEY) or
        (CAPTCHA_PROVIDER == "recaptcha" and RECAPTCHA_SECRET_KEY)
    )


def set_captcha_enabled(enabled: bool) -> None:
    """
    Enable or disable CAPTCHA at runtime (admin only).
    This does not persist across restarts - use CAPTCHA_PROVIDER=none for permanent disable.
    """
    global _captcha_runtime_enabled
    _captcha_runtime_enabled = enabled
    logger.info(f"CAPTCHA runtime toggle set to: {enabled}")


def get_captcha_runtime_status() -> bool:
    """Get current runtime status of CAPTCHA toggle."""
    return _captcha_runtime_enabled
