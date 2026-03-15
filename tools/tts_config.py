"""
ElevenLabs TTS configuration loader.
Reads per-profile settings from SYSTEM_CONFIG with TTL cache.
"""
import time
import orjson
from database import get_db_connection
from log_config import logger

# Cache TTL
_TTS_CONFIG_CACHE_TTL = 300  # 5 minutes
_tts_config_cache: dict = {}
_tts_config_cache_time: float = 0

# Hardcoded fallbacks (match pre-config behavior)
_DEFAULTS = {
    "tts_webchat_model":          "eleven_turbo_v2_5",
    "tts_webchat_format":         "mp3_44100_128",
    "tts_webchat_stability":      "0.45",
    "tts_webchat_similarity":     "0.89",
    "tts_webchat_ws_enabled":     "1",
    "tts_webchat_chunk_schedule": "[120, 160, 250, 290]",

    "tts_external_model":         "eleven_turbo_v2_5",
    "tts_external_format":        "opus_48000_128",
    "tts_external_stability":     "0.45",
    "tts_external_similarity":    "0.89",

    "tts_mp3_model":              "eleven_turbo_v2_5",
    "tts_mp3_format":             "opus_48000_128",
    "tts_mp3_stability":          "0.45",
    "tts_mp3_similarity":         "0.89",
}

# Valid models for admin page dropdowns and validation
VALID_MODELS = [
    ("eleven_flash_v2_5",      "Flash v2.5 -- Fastest, 50% cheaper, good quality"),
    ("eleven_turbo_v2_5",      "Turbo v2.5 -- Low latency, very good quality (current)"),
    ("eleven_multilingual_v2", "Multilingual v2 -- Best quality, 29 languages, higher latency"),
    ("eleven_v3",              "v3 -- Best quality, highest latency (NO WebSocket support)"),
]

# Models that do NOT support WebSocket TTS API
WS_INCOMPATIBLE_MODELS = {"eleven_v3"}

# Valid formats: (value, description, pydub_format)
VALID_FORMATS = [
    ("mp3_22050_32",   "MP3 22kHz 32kbps -- Low quality, tiny size",               "mp3"),
    ("mp3_44100_64",   "MP3 44kHz 64kbps -- Medium quality",                       "mp3"),
    ("mp3_44100_96",   "MP3 44kHz 96kbps -- Good quality",                         "mp3"),
    ("mp3_44100_128",  "MP3 44kHz 128kbps -- High quality (recommended streaming)", "mp3"),
    ("mp3_44100_192",  "MP3 44kHz 192kbps -- Very high quality",                   "mp3"),
    ("opus_48000_128", "Opus 48kHz 128kbps -- Efficient, great for messaging",     "ogg"),
]

# Lookup for pydub format mapping (single source of truth)
_FORMAT_TO_PYDUB = {f[0]: f[2] for f in VALID_FORMATS}

# Valid profile names
VALID_PROFILES = {"webchat", "external", "mp3"}


class TTSProfile:
    """Parsed TTS settings for a specific context."""
    __slots__ = ("model_id", "output_format", "stability", "similarity_boost",
                 "ws_enabled", "chunk_schedule")

    def __init__(self, model_id: str, output_format: str, stability: float,
                 similarity_boost: float, ws_enabled: bool = False,
                 chunk_schedule: list | None = None):
        self.model_id = model_id
        self.output_format = output_format
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.ws_enabled = ws_enabled
        self.chunk_schedule = chunk_schedule or [120, 160, 250, 290]


def format_to_pydub(output_format: str) -> str:
    """Map ElevenLabs output_format to pydub input format string.
    Uses the lookup table from VALID_FORMATS (single source of truth)."""
    return _FORMAT_TO_PYDUB.get(output_format, "ogg")


def invalidate_tts_config_cache():
    """Call after saving new TTS settings to force immediate reload."""
    global _tts_config_cache, _tts_config_cache_time
    _tts_config_cache = {}
    _tts_config_cache_time = 0


async def get_tts_config() -> dict:
    """Return raw key-value dict of all tts_* SYSTEM_CONFIG keys. Cached for 5 min."""
    global _tts_config_cache, _tts_config_cache_time
    now = time.time()
    if _tts_config_cache and (now - _tts_config_cache_time) < _TTS_CONFIG_CACHE_TTL:
        return _tts_config_cache

    config = dict(_DEFAULTS)  # start with hardcoded fallbacks
    try:
        async with get_db_connection(readonly=True) as conn:
            async with conn.execute(
                "SELECT key, value FROM SYSTEM_CONFIG WHERE key LIKE 'tts_%'"
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                config[row["key"]] = row["value"]
    except Exception as e:
        logger.error("Failed to load TTS config from DB, using defaults: %s", e)

    _tts_config_cache = config
    _tts_config_cache_time = now
    return config


async def get_tts_profile(context: str) -> TTSProfile:
    """Get parsed TTS settings for a context: 'webchat', 'external', or 'mp3'."""
    if context not in VALID_PROFILES:
        raise ValueError(f"Unknown TTS context: {context}")

    config = await get_tts_config()
    prefix = f"tts_{context}_"

    model_id = config.get(f"{prefix}model", _DEFAULTS.get(f"{prefix}model", "eleven_turbo_v2_5"))
    output_format = config.get(f"{prefix}format", _DEFAULTS.get(f"{prefix}format", "opus_48000_128"))

    try:
        stability = float(config.get(f"{prefix}stability", "0.45"))
    except (ValueError, TypeError):
        stability = 0.45

    try:
        similarity = float(config.get(f"{prefix}similarity", "0.89"))
    except (ValueError, TypeError):
        similarity = 0.89

    ws_enabled = False
    chunk_schedule = [120, 160, 250, 290]

    if context == "webchat":
        ws_enabled = config.get(f"{prefix}ws_enabled", "1") == "1"
        try:
            chunk_schedule = orjson.loads(
                config.get(f"{prefix}chunk_schedule", "[120, 160, 250, 290]")
            )
        except Exception:
            chunk_schedule = [120, 160, 250, 290]

    return TTSProfile(
        model_id=model_id,
        output_format=output_format,
        stability=stability,
        similarity_boost=similarity,
        ws_enabled=ws_enabled,
        chunk_schedule=chunk_schedule,
    )
