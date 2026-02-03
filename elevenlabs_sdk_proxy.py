"""Proxy helpers to serve the ElevenLabs ConvAI SDK from local endpoints."""

import os
from typing import List, Optional

import httpx
from fastapi import HTTPException

ELEVEN_CLIENT_VER = (os.getenv("ELEVEN_CLIENT_VER", "0.5.2") or "0.5.2").strip()

ELEVEN_CLIENT_URLS: List[str] = [
    f"https://cdn.jsdelivr.net/npm/@elevenlabs/client@{ELEVEN_CLIENT_VER}/dist/lib.umd.js",
    f"https://unpkg.com/@elevenlabs/client@{ELEVEN_CLIENT_VER}/dist/lib.umd.js",
]


class ElevenLabsSDKProxy:
    """Downloads and caches ElevenLabs ConvAI SDK assets."""

    _sdk_cache: Optional[bytes] = None
    _sourcemap_cache: Optional[bytes] = None

    @classmethod
    async def get_sdk(cls) -> bytes:
        """Return the ElevenLabs SDK JavaScript bundle."""
        if cls._sdk_cache:
            return cls._sdk_cache
        cls._sdk_cache = await cls._download(ELEVEN_CLIENT_URLS)
        return cls._sdk_cache

    @classmethod
    async def get_sourcemap(cls) -> bytes:
        """Return the ElevenLabs SDK sourcemap if available."""
        if cls._sourcemap_cache:
            return cls._sourcemap_cache

        map_urls: List[str] = []
        for url in ELEVEN_CLIENT_URLS:
            base = url.rsplit("/", 1)[0]
            map_urls.append(f"{base}/lib.umd.js.map")

        cls._sourcemap_cache = await cls._download(map_urls)
        return cls._sourcemap_cache

    @classmethod
    async def _download(cls, urls: List[str]) -> bytes:
        errors: List[str] = []
        async with httpx.AsyncClient(timeout=20.0) as client:
            for url in urls:
                try:
                    response = await client.get(url)
                    if response.status_code == 200 and response.content:
                        return response.content
                    errors.append(f"{url} -> {response.status_code}")
                except Exception as exc:  # pragma: no cover - network errors
                    errors.append(f"{url} -> {exc}")
        raise HTTPException(status_code=502, detail="Could not download ElevenLabs SDK: " + " | ".join(errors))
