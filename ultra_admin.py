"""
Ultra Admin+ — Temporary privilege elevation for admin users.

Allows admins to verify their identity via email code and temporarily
gain elevated privileges (e.g., deleting other admin accounts).
All state is stored in Redis with automatic TTL expiration.
"""

import secrets
import json
import time
from rediscfg import redis_client
from log_config import logger


# TTL constants (seconds)
CODE_TTL = 300              # 5 min — verification code validity
ELEVATION_TTL = 1800        # 30 min — elevated session duration
MAX_CODE_ATTEMPTS = 3       # Max wrong code entries before code is burned
REQUEST_COOLDOWN_TTL = 60   # 1 min — cooldown between code requests


async def generate_elevation_code(user_id: int) -> str | None:
    """
    Generate a 6-digit verification code and store in Redis.
    Returns the code string, or None if on cooldown.
    """
    cooldown_key = f"ultra_admin:cooldown:{user_id}"
    cooldown_set = await redis_client.set(cooldown_key, 1, nx=True, ex=REQUEST_COOLDOWN_TTL)
    if not cooldown_set:
        return None

    code = f"{secrets.randbelow(1000000):06d}"
    await redis_client.setex(f"ultra_admin:code:{user_id}", CODE_TTL, code)
    await redis_client.delete(f"ultra_admin:attempts:{user_id}")
    return code


async def get_active_lock_owner() -> int | None:
    """
    Check if another admin is currently elevated.
    Returns the elevated user_id, or None if nobody is elevated.
    """
    lock_val = await redis_client.get("ultra_admin:active_lock")
    if not lock_val:
        return None
    return int(lock_val)


async def verify_elevation_code(user_id: int, submitted_code: str, ip_address: str) -> tuple[bool, str]:
    """
    Verify the submitted code. Returns (success, message).
    On success, sets the elevation state in Redis.
    Uses secrets.compare_digest for timing-safe comparison.
    """
    code_key = f"ultra_admin:code:{user_id}"
    attempts_key = f"ultra_admin:attempts:{user_id}"

    stored_code = await redis_client.get(code_key)
    if not stored_code:
        return False, "no_code"

    # Check attempts
    raw_attempts = await redis_client.get(attempts_key)
    current_attempts = int(raw_attempts) if raw_attempts else 0

    if current_attempts >= MAX_CODE_ATTEMPTS:
        await redis_client.delete(code_key)
        return False, "max_attempts"

    if not secrets.compare_digest(submitted_code, stored_code):
        await redis_client.incr(attempts_key)
        await redis_client.expire(attempts_key, 900)
        remaining = MAX_CODE_ATTEMPTS - current_attempts - 1
        return False, f"wrong_code:{remaining}"

    # Code is correct — try to acquire lock atomically BEFORE deleting the code
    # Check if lock belongs to another user
    lock_owner = await get_active_lock_owner()
    if lock_owner is not None and lock_owner != user_id:
        return False, "already_elevated"

    # Atomic lock acquisition: only succeeds if no lock exists (NX)
    lock_acquired = await redis_client.set(
        "ultra_admin:active_lock", str(user_id),
        nx=True, ex=ELEVATION_TTL
    )

    # If lock exists but belongs to us (re-elevation), force set it
    if not lock_acquired:
        current_lock = await get_active_lock_owner()
        if current_lock == user_id:
            await redis_client.setex("ultra_admin:active_lock", ELEVATION_TTL, str(user_id))
        else:
            return False, "already_elevated"

    # Lock acquired — now clean up code and set elevation
    await redis_client.delete(code_key)
    await redis_client.delete(attempts_key)

    elevation_data = json.dumps({
        "elevated_at": int(time.time()),
        "ip": ip_address,
        "user_id": user_id
    })
    await redis_client.setex(f"ultra_admin:elevated:{user_id}", ELEVATION_TTL, elevation_data)

    return True, "elevated"


async def is_elevated(user_id: int, request_ip: str = None) -> bool:
    """
    Check if user has Ultra Admin+ elevation.
    If request_ip is provided, also verifies IP matches the elevation IP.
    """
    data = await redis_client.get(f"ultra_admin:elevated:{user_id}")
    if not data:
        return False

    if request_ip:
        elevation = json.loads(data)
        if elevation.get("ip") != request_ip:
            logger.warning(
                f"[ULTRA ADMIN+] IP mismatch for user {user_id}: "
                f"elevation IP={elevation.get('ip')}, request IP={request_ip}"
            )
            return False

    return True


async def revoke_elevation(user_id: int):
    """Revoke Ultra Admin+ elevation immediately."""
    await redis_client.delete(f"ultra_admin:elevated:{user_id}")
    # Clean lock if it belongs to this user
    lock_val = await redis_client.get("ultra_admin:active_lock")
    if lock_val:
        if int(lock_val) == user_id:
            await redis_client.delete("ultra_admin:active_lock")


async def get_elevation_ttl(user_id: int) -> int:
    """Get remaining seconds for elevation. Returns -1 if not elevated."""
    ttl = await redis_client.ttl(f"ultra_admin:elevated:{user_id}")
    return ttl if ttl and ttl > 0 else -1
