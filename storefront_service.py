"""Service layer for creator storefronts — data access and business logic."""

import re
from typing import Optional

import orjson

from common import get_manager_branding, slugify
from database import get_db_connection
from log_config import logger
from security_config import is_forbidden_prompt_name


# ---------------------------------------------------------------------------
# Social link validation
# ---------------------------------------------------------------------------

ALLOWED_SOCIAL_KEYS = {'website', 'twitter', 'linkedin', 'youtube', 'instagram', 'github'}
URL_PATTERN = re.compile(r'^https?://.+', re.IGNORECASE)


def validate_social_links(social_links: dict) -> dict:
    """Return a cleaned copy containing only valid social link entries."""
    if not isinstance(social_links, dict):
        return {}
    cleaned = {}
    for key, value in social_links.items():
        if key not in ALLOWED_SOCIAL_KEYS:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        # Block javascript: and other dangerous schemes
        if value.lower().startswith('javascript:'):
            continue
        if not URL_PATTERN.match(value):
            continue
        if len(value) > 500:
            continue
        cleaned[key] = value
    return cleaned


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

async def generate_unique_creator_slug(
    display_name: str,
    exclude_user_id: Optional[int] = None,
) -> str:
    """Create a unique slug for a creator profile based on *display_name*.

    Uses ``slugify()`` from common.py, then checks CREATOR_PROFILES for
    collisions — appending ``-1``, ``-2``, etc. up to 100 tries.
    *exclude_user_id* allows the current user to keep their own slug when
    updating their profile.
    """
    base = slugify(display_name)
    if not base:
        base = "creator"

    if is_forbidden_prompt_name(base):
        base = f"creator-{base}"

    candidate = base
    async with get_db_connection(readonly=True) as conn:
        for i in range(101):  # 0 = original, 1..100 = suffixed
            if i > 0:
                candidate = f"{base}-{i}"

            if exclude_user_id is not None:
                cursor = await conn.execute(
                    "SELECT 1 FROM CREATOR_PROFILES WHERE slug = ? AND user_id != ?",
                    (candidate, exclude_user_id),
                )
            else:
                cursor = await conn.execute(
                    "SELECT 1 FROM CREATOR_PROFILES WHERE slug = ?",
                    (candidate,),
                )

            if not await cursor.fetchone():
                return candidate

    # Exhausted all attempts — should never happen in practice
    return f"{base}-{int(orjson.dumps(display_name).hex(), 16) % 10000}"


# ---------------------------------------------------------------------------
# Profile queries
# ---------------------------------------------------------------------------

_PROFILE_SELECT = """
    SELECT cp.user_id, cp.slug, cp.display_name, cp.bio, cp.avatar_url,
           cp.social_links, cp.branding_id, cp.is_public, cp.is_verified,
           cp.created_at, u.username
    FROM CREATOR_PROFILES cp
    JOIN USERS u ON cp.user_id = u.id
"""


def _row_to_profile(row) -> dict:
    """Map a raw row from the profile SELECT to a dict."""
    return {
        'user_id': row[0],
        'slug': row[1],
        'display_name': row[2],
        'bio': row[3],
        'avatar_url': row[4],
        'social_links': orjson.loads(row[5]) if row[5] else {},
        'branding_id': row[6],
        'is_public': bool(row[7]),
        'is_verified': bool(row[8]),
        'created_at': row[9],
        'username': row[10],
    }


async def get_creator_profile_by_slug(slug: str) -> Optional[dict]:
    """Return the public creator profile for *slug*, or ``None``."""
    async with get_db_connection(readonly=True) as conn:
        cursor = await conn.execute(
            _PROFILE_SELECT + " WHERE cp.slug = ? AND cp.is_public = 1",
            (slug,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return _row_to_profile(row)


async def get_own_creator_profile(user_id: int) -> Optional[dict]:
    """Return the creator profile for *user_id* regardless of public status.

    Used on the management / settings page so the creator can see and edit
    their own profile even before publishing it.
    """
    async with get_db_connection(readonly=True) as conn:
        cursor = await conn.execute(
            _PROFILE_SELECT + " WHERE cp.user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return _row_to_profile(row)


# ---------------------------------------------------------------------------
# Auto-creation hook
# ---------------------------------------------------------------------------

async def ensure_creator_profile(user_id: int, username: str) -> bool:
    """Auto-create a creator profile if one does not exist.

    Called from the prompt-creation flow so every creator gets a profile row.
    The profile starts as ``is_public=0`` (draft).

    Returns ``True`` if a new row was inserted, ``False`` if it already existed.
    """
    try:
        async with get_db_connection() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM CREATOR_PROFILES WHERE user_id = ?", (user_id,)
            )
            if await cursor.fetchone():
                return False

            slug = await generate_unique_creator_slug(username)

            await conn.execute(
                """
                INSERT OR IGNORE INTO CREATOR_PROFILES (user_id, slug, display_name, is_public)
                VALUES (?, ?, ?, 0)
                """,
                (user_id, slug, username),
            )
            await conn.commit()
            logger.info(
                "Auto-created creator profile for user %d with slug '%s'",
                user_id,
                slug,
            )
            return True
    except Exception as e:
        logger.warning(
            "Failed to auto-create creator profile for user %d: %s", user_id, e
        )
        return False


# ---------------------------------------------------------------------------
# Full storefront assembly
# ---------------------------------------------------------------------------

async def get_creator_storefront_data(
    creator_user_id: int,
    viewer_user_id: Optional[int] = None,
) -> Optional[dict]:
    """Assemble every piece of data needed to render a creator storefront.

    Returns a dict with keys: profile, branding, avatar_url, prompts, packs,
    stats, viewer_access.  Returns ``None`` if the creator has no profile row.
    """
    async with get_db_connection(readonly=True) as conn:
        # 1. Creator profile ------------------------------------------------
        cursor = await conn.execute(
            _PROFILE_SELECT + " WHERE cp.user_id = ?",
            (creator_user_id,),
        )
        profile_row = await cursor.fetchone()
        if not profile_row:
            return None

        profile = _row_to_profile(profile_row)

        # 2. Branding -------------------------------------------------------
        branding = await get_manager_branding(creator_user_id, conn)

        # 3. Avatar URL -----------------------------------------------------
        avatar_url = profile['avatar_url'] if profile['avatar_url'] else None

        # 4. Public prompts owned by this creator ---------------------------
        cursor = await conn.execute(
            """
            SELECT p.id, p.name, p.description, p.image, p.public_id,
                   p.is_paid, p.markup_per_mtokens
            FROM PROMPTS p
            JOIN PROMPT_PERMISSIONS pp ON p.id = pp.prompt_id
            WHERE pp.user_id = ? AND pp.permission_level = 'owner'
              AND p.public = 1 AND p.is_unlisted = 0
            ORDER BY p.created_at DESC
            """,
            (creator_user_id,),
        )
        prompt_rows = await cursor.fetchall()

        prompts = [
            {
                'id': r[0],
                'name': r[1],
                'description': r[2],
                'image': r[3],
                'public_id': r[4],
                'is_paid': bool(r[5]),
                'markup_per_mtokens': r[6],
            }
            for r in prompt_rows
        ]

        # 5. Published packs owned by this creator -------------------------
        cursor = await conn.execute(
            """
            SELECT pk.id, pk.name, pk.slug, pk.description, pk.cover_image,
                   pk.is_paid, pk.price, pk.public_id,
                   (SELECT COUNT(*) FROM PACK_ITEMS pi
                    WHERE pi.pack_id = pk.id AND pi.is_active = 1) AS item_count
            FROM PACKS pk
            WHERE pk.created_by_user_id = ? AND pk.status = 'published'
              AND pk.is_public = 1
            ORDER BY pk.created_at DESC
            """,
            (creator_user_id,),
        )
        pack_rows = await cursor.fetchall()

        packs = [
            {
                'id': r[0],
                'name': r[1],
                'slug': r[2],
                'description': r[3],
                'cover_image': r[4],
                'is_paid': bool(r[5]),
                'price': r[6],
                'public_id': r[7],
                'item_count': r[8],
            }
            for r in pack_rows
        ]

        # 6. Viewer access (if a logged-in user is browsing) ---------------
        viewer_access: dict = {'prompt_ids': [], 'pack_ids': []}
        if viewer_user_id:
            prompt_ids = [p['id'] for p in prompts]
            if prompt_ids:
                placeholders = ','.join('?' * len(prompt_ids))
                cursor = await conn.execute(
                    f"""
                    SELECT DISTINCT prompt_id FROM PROMPT_PERMISSIONS
                    WHERE user_id = ? AND prompt_id IN ({placeholders})
                    """,
                    (viewer_user_id, *prompt_ids),
                )
                rows = await cursor.fetchall()
                viewer_access['prompt_ids'] = [r[0] for r in rows]

            pack_ids = [p['id'] for p in packs]
            if pack_ids:
                placeholders = ','.join('?' * len(pack_ids))
                cursor = await conn.execute(
                    f"""
                    SELECT pack_id FROM PACK_ACCESS
                    WHERE user_id = ? AND pack_id IN ({placeholders})
                    """,
                    (viewer_user_id, *pack_ids),
                )
                rows = await cursor.fetchall()
                viewer_access['pack_ids'] = [r[0] for r in rows]

        return {
            'profile': profile,
            'branding': branding,
            'avatar_url': avatar_url,
            'prompts': prompts,
            'packs': packs,
            'stats': {
                'prompt_count': len(prompts),
                'pack_count': len(packs),
            },
            'viewer_access': viewer_access,
        }
