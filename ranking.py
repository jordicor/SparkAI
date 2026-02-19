# ranking.py
# Explore Ranking Engine - scores prompts and packs for the /explore page

import time
import json
import asyncio
from datetime import datetime, timezone

from database import get_db_connection
from log_config import logger

# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "users_with_access": 3,
    "purchases": 5,
    "unique_chatters": 4,
    "landing_conversions": 6,
    "favorites": 2,
    "has_landing_boost": 15,
    "recency_max_bonus": 30,
}

RECENCY_WINDOW_DAYS = 90

# ---------------------------------------------------------------------------
# Config cache (same pattern as pricing config in common.py)
# ---------------------------------------------------------------------------
_ranking_config_cache: dict = {}
_ranking_config_cache_time: float = 0
RANKING_CONFIG_CACHE_TTL = 300  # 5 minutes


async def get_ranking_config() -> dict:
    """Return ranking configuration from SYSTEM_CONFIG, cached for 5 min."""
    global _ranking_config_cache, _ranking_config_cache_time

    now = time.time()
    if _ranking_config_cache and (now - _ranking_config_cache_time) < RANKING_CONFIG_CACHE_TTL:
        return _ranking_config_cache

    config: dict = {
        "mode": "piggyback",
        "interval_hours": 6,
        "weights": DEFAULT_WEIGHTS.copy(),
        "last_updated": 0.0,
    }

    async with get_db_connection(readonly=True) as conn:
        cursor = await conn.execute(
            "SELECT key, value FROM SYSTEM_CONFIG WHERE key LIKE 'ranking_%'"
        )
        rows = await cursor.fetchall()

    for row in rows:
        key, value = row[0], row[1]
        if key == "ranking_mode":
            config["mode"] = value
        elif key == "ranking_interval_hours":
            config["interval_hours"] = int(value)
        elif key == "ranking_weights":
            config["weights"] = json.loads(value)
        elif key == "ranking_last_updated":
            config["last_updated"] = float(value)

    _ranking_config_cache = config
    _ranking_config_cache_time = now
    return config


def invalidate_ranking_config_cache() -> None:
    """Reset cache so next call to get_ranking_config() hits the DB."""
    global _ranking_config_cache, _ranking_config_cache_time
    _ranking_config_cache = {}
    _ranking_config_cache_time = 0


# ---------------------------------------------------------------------------
# Core recalculation
# ---------------------------------------------------------------------------
_recalculation_running = False

PROMPT_SIGNALS_SQL = """
SELECT p.id, p.has_landing_page, p.created_at,
    COALESCE(pp.cnt, 0),
    COALESCE(pur.cnt, 0),
    COALESCE(ch.cnt, 0),
    COALESCE(cv.cnt, 0),
    COALESCE(fv.cnt, 0)
FROM PROMPTS p
LEFT JOIN (SELECT prompt_id, COUNT(*) cnt FROM PROMPT_PERMISSIONS WHERE permission_level='access' GROUP BY prompt_id) pp ON pp.prompt_id = p.id
LEFT JOIN (SELECT prompt_id, COUNT(*) cnt FROM PROMPT_PURCHASES GROUP BY prompt_id) pur ON pur.prompt_id = p.id
LEFT JOIN (SELECT role_id, COUNT(DISTINCT user_id) cnt FROM CONVERSATIONS GROUP BY role_id) ch ON ch.role_id = p.id
LEFT JOIN (SELECT prompt_id, COUNT(*) cnt FROM LANDING_PAGE_ANALYTICS WHERE converted=1 GROUP BY prompt_id) cv ON cv.prompt_id = p.id
LEFT JOIN (SELECT prompt_id, COUNT(*) cnt FROM FAVORITE_PROMPTS GROUP BY prompt_id) fv ON fv.prompt_id = p.id
WHERE p.public = 1 AND p.is_unlisted = 0
"""

PACK_SIGNALS_SQL = """
SELECT pk.id, pk.has_custom_landing, pk.created_at,
    COALESCE(pa.cnt, 0),
    COALESCE(cv.cnt, 0)
FROM PACKS pk
LEFT JOIN (SELECT pack_id, COUNT(*) cnt FROM PACK_ACCESS GROUP BY pack_id) pa ON pa.pack_id = pk.id
LEFT JOIN (SELECT pack_id, COUNT(*) cnt FROM LANDING_PAGE_ANALYTICS WHERE converted=1 AND pack_id IS NOT NULL GROUP BY pack_id) cv ON cv.pack_id = pk.id
WHERE pk.status = 'published' AND pk.is_public = 1
"""


def _days_since(created_at_str: str) -> float:
    """Return number of days between created_at and now (UTC)."""
    try:
        created_dt = datetime.fromisoformat(created_at_str)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created_dt).total_seconds() / 86400
    except (ValueError, TypeError):
        return RECENCY_WINDOW_DAYS


def _recency_bonus(days_old: float, weight: float) -> float:
    """Linear decay over RECENCY_WINDOW_DAYS, floored at 0."""
    return max(0.0, weight * (1 - days_old / RECENCY_WINDOW_DAYS))


async def recalculate_ranking_scores() -> None:
    """Recalculate ranking_score for all public prompts and published packs."""
    global _recalculation_running
    _recalculation_running = True
    t0 = time.time()
    logger.info("Ranking recalculation started")

    try:
        config = await get_ranking_config()
        w = config["weights"]
        w1 = w.get("users_with_access", 3)
        w2 = w.get("purchases", 5)
        w3 = w.get("unique_chatters", 4)
        w4 = w.get("landing_conversions", 6)
        w5 = w.get("favorites", 2)
        w6 = w.get("has_landing_boost", 15)
        w7 = w.get("recency_max_bonus", 30)

        # -- Prompts ----------------------------------------------------------
        async with get_db_connection(readonly=True) as conn:
            cursor = await conn.execute(PROMPT_SIGNALS_SQL)
            prompt_rows = await cursor.fetchall()

            cursor = await conn.execute(PACK_SIGNALS_SQL)
            pack_rows = await cursor.fetchall()

        prompt_updates = []
        for row in prompt_rows:
            pid, has_landing, created_at, access_cnt, purchase_cnt, chatter_cnt, conv_cnt, fav_cnt = row
            days_old = _days_since(created_at) if created_at else RECENCY_WINDOW_DAYS
            score = (
                access_cnt * w1
                + purchase_cnt * w2
                + chatter_cnt * w3
                + conv_cnt * w4
                + fav_cnt * w5
                + (w6 if has_landing else 0)
                + _recency_bonus(days_old, w7)
            )
            prompt_updates.append((round(score, 2), pid))

        pack_updates = []
        for row in pack_rows:
            pk_id, has_landing, created_at, access_cnt, conv_cnt = row
            days_old = _days_since(created_at) if created_at else RECENCY_WINDOW_DAYS
            score = (
                access_cnt * w1
                + conv_cnt * w4
                + (w6 if has_landing else 0)
                + _recency_bonus(days_old, w7)
            )
            pack_updates.append((round(score, 2), pk_id))

        # -- Write scores & timestamp -----------------------------------------
        async with get_db_connection() as conn:
            if prompt_updates:
                await conn.executemany(
                    "UPDATE PROMPTS SET ranking_score = ? WHERE id = ?",
                    prompt_updates,
                )
            if pack_updates:
                await conn.executemany(
                    "UPDATE PACKS SET ranking_score = ? WHERE id = ?",
                    pack_updates,
                )
            now_ts = str(time.time())
            await conn.execute(
                "UPDATE SYSTEM_CONFIG SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = 'ranking_last_updated'",
                (now_ts,),
            )
            await conn.commit()

        invalidate_ranking_config_cache()
        elapsed = round(time.time() - t0, 3)
        logger.info("Ranking recalculation done: %d prompts, %d packs in %ss",
                     len(prompt_updates), len(pack_updates), elapsed)
    finally:
        _recalculation_running = False


# ---------------------------------------------------------------------------
# Piggyback trigger (fire-and-forget from explore endpoints)
# ---------------------------------------------------------------------------
async def maybe_trigger_recalculation() -> None:
    """If enough time has passed, fire-and-forget a ranking recalculation."""
    global _recalculation_running
    if _recalculation_running:
        return

    config = await get_ranking_config()
    elapsed = time.time() - config["last_updated"]
    if elapsed < config["interval_hours"] * 3600:
        return

    _recalculation_running = True
    asyncio.create_task(recalculate_ranking_scores())


# ---------------------------------------------------------------------------
# Scheduled loop (alternative to piggyback, started from app lifespan)
# ---------------------------------------------------------------------------
async def start_scheduled_ranking_loop() -> None:
    """Run recalculate_ranking_scores() on a fixed interval forever."""
    while True:
        try:
            await recalculate_ranking_scores()
        except Exception:
            logger.exception("Scheduled ranking recalculation failed")
        config = await get_ranking_config()
        await asyncio.sleep(config["interval_hours"] * 3600)
