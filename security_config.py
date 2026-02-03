"""
Security Configuration Module.

Loads forbidden names from JSON file ONCE at import time.
The data stays in memory for the lifetime of the application.

Usage:
    from security_config import is_forbidden_prompt_name, is_forbidden_username

    if is_forbidden_prompt_name("wp-admin"):
        # Block this name
"""

import os
import orjson
import logging
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

# =============================================================================
# Load forbidden names from JSON (executed once at import)
# =============================================================================

_CONFIG_PATH = Path(__file__).parent / "data" / "config" / "forbidden_names.json"

# Sets for O(1) lookup
_forbidden_prompt_names: Set[str] = set()
_forbidden_username_names: Set[str] = set()


def _load_forbidden_names():
    """Load forbidden names from JSON file into memory."""
    global _forbidden_prompt_names, _forbidden_username_names

    if not _CONFIG_PATH.exists():
        logger.warning(f"Forbidden names config not found: {_CONFIG_PATH}")
        # Use hardcoded fallback for critical names
        _forbidden_prompt_names = {
            "wp-admin", "wp-content", "phpmyadmin", "admin", "credentials",
            "secrets", "docker", "kubernetes", "internal", "hidden", "debug"
        }
        _forbidden_username_names = {
            "admin", "administrator", "root", "system", "support"
        }
        return

    try:
        with open(_CONFIG_PATH, "rb") as f:
            config = orjson.loads(f.read())

        # Load and normalize to lowercase for case-insensitive matching
        _forbidden_prompt_names = {
            name.lower().strip()
            for name in config.get("forbidden_prompt_names", [])
            if name and isinstance(name, str)
        }

        _forbidden_username_names = {
            name.lower().strip()
            for name in config.get("forbidden_username_names", [])
            if name and isinstance(name, str)
        }

        logger.info(
            f"Loaded forbidden names: {len(_forbidden_prompt_names)} prompt names, "
            f"{len(_forbidden_username_names)} username names"
        )

    except orjson.JSONDecodeError as e:
        logger.error(f"Error parsing forbidden names JSON: {e}")
        # Use minimal fallback
        _forbidden_prompt_names = {"admin", "wp-admin", "phpmyadmin"}
        _forbidden_username_names = {"admin", "root"}

    except Exception as e:
        logger.error(f"Error loading forbidden names: {e}")
        _forbidden_prompt_names = {"admin", "wp-admin", "phpmyadmin"}
        _forbidden_username_names = {"admin", "root"}


# Load on import
_load_forbidden_names()


# =============================================================================
# Public API
# =============================================================================

def is_forbidden_prompt_name(name: str) -> bool:
    """
    Check if a prompt name is forbidden.

    Args:
        name: The prompt name to check (case-insensitive)

    Returns:
        True if the name is forbidden, False otherwise
    """
    if not name:
        return False

    # Normalize: lowercase, strip whitespace
    normalized = name.lower().strip()

    # Also check the slugified version (hyphens instead of spaces)
    slugified = normalized.replace(" ", "-").replace("_", "-")

    return normalized in _forbidden_prompt_names or slugified in _forbidden_prompt_names


def is_forbidden_username(name: str) -> bool:
    """
    Check if a username is forbidden.

    Args:
        name: The username to check (case-insensitive)

    Returns:
        True if the username is forbidden, False otherwise
    """
    if not name:
        return False

    normalized = name.lower().strip()
    return normalized in _forbidden_username_names


def get_forbidden_prompt_names() -> Set[str]:
    """Get a copy of all forbidden prompt names (for debugging/admin)."""
    return _forbidden_prompt_names.copy()


def get_forbidden_username_names() -> Set[str]:
    """Get a copy of all forbidden username names (for debugging/admin)."""
    return _forbidden_username_names.copy()


def reload_forbidden_names():
    """
    Reload forbidden names from the JSON file.
    Useful if the config was updated without restarting the app.
    """
    _load_forbidden_names()
    logger.info("Forbidden names reloaded from config file")
