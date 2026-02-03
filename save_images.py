# save_images.py

import os
import io
import hashlib
import redis
import aiosqlite
from typing import Optional
import jwt
from jwt import PyJWTError as JWTError
from urllib.parse import urlparse
from datetime import date, datetime, timezone, timedelta
from PIL import Image as PilImage, UnidentifiedImageError
from fastapi import FastAPI, Response, HTTPException, Depends, Request, Form, status, UploadFile, File
from dotenv import load_dotenv

# own libraries
from models import User, ConnectionManager
from log_config import logger
from auth import hash_password, verify_password, get_user_by_username, get_current_user, create_access_token, get_user_by_id, get_user_from_phone_number
from auth import get_current_user_from_websocket, get_user_id_from_conversation, get_user_by_token, create_user_info, create_login_response, generate_magic_link
from common import CLOUDFLARE_FOR_IMAGES, CLOUDFLARE_IMAGE_SUBDOMAIN, CLOUDFLARE_BASE_URL, generate_signed_url_cloudflare
from common import Cost, generate_user_hash, has_sufficient_balance, cost_tts, cache_directory, users_directory, elevenlabs_key, openai_key, tts_engine, get_balance, deduct_balance, load_service_costs, SECRET_KEY, ALGORITHM, MEDIA_TOKEN_EXPIRE_HOURS

# Load environment variables
load_dotenv()

# Token storage configuration
USE_REDIS = os.getenv('REDIS_IMG_TOKEN', '0') == '1'

# Variables globales
redis_client = None
conn_mem = None

if USE_REDIS:
    # Initialize Redis
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True
    )

async def save_image_locally(
    request: Optional[Request],
    image_data: bytes,
    current_user,
    conversation_id: int,
    filename: str,
    source: str,
    format: str = 'webp',
    scheme: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None
) -> tuple:
    # Generate user hash
    hash_prefix1, hash_prefix2, user_hash = generate_user_hash(current_user.username)
    
    # Create the conversation prefixes according to the specified format
    conversation_id_str = f"{conversation_id:07d}"
    conversation_id_prefix1 = conversation_id_str[:3]
    conversation_id_prefix2 = conversation_id_str[3:]
    
    # Build directory path
    file_location = os.path.join(users_directory, hash_prefix1, hash_prefix2, user_hash, "files", conversation_id_prefix1, conversation_id_prefix2, "img", source)
    
    # Create directory if it does not exist
    if not os.path.exists(file_location):
        os.makedirs(file_location)

    # Load image from received bytes
    image = PilImage.open(io.BytesIO(image_data))

    # Check if the image is smaller than 256x256
    width, height = image.size
    if width <= 256 and height <= 256:
        # If the image is smaller, we don't resize it
        image_256 = image
    else:
        # If it's larger, we resize while maintaining the aspect ratio
        image_256 = resize_image(image, 256)

    # Generate unique file names
    file_hash = hashlib.sha1(image_data).hexdigest()
    
    # Use specified format
    ext = format.lower()
    
    filename_256 = f"{file_hash}_256.{ext}"
    filename_fullsize = f"{file_hash}_fullsize.{ext}"

    file_path_256 = os.path.join(file_location, filename_256)
    file_path_fullsize = os.path.join(file_location, filename_fullsize)

    # Save image (resized or not) as _256
    image_256.save(file_path_256, ext.upper())

    # Save original image as fullsize
    image.save(file_path_fullsize, ext.upper())
    
    base_url_256 = f"users/{hash_prefix1}/{hash_prefix2}/{user_hash}/files/{conversation_id_prefix1}/{conversation_id_prefix2}/img/{source}/{filename_256}"
    base_url_fullsize = f"users/{hash_prefix1}/{hash_prefix2}/{user_hash}/files/{conversation_id_prefix1}/{conversation_id_prefix2}/img/{source}/{filename_fullsize}"

    if CLOUDFLARE_FOR_IMAGES:
        # Generate signed Cloudflare URLs
        image_link_token_256 = generate_signed_url_cloudflare(base_url_256, expiration_seconds=3600)
        image_link_token_fullsize = generate_signed_url_cloudflare(base_url_fullsize, expiration_seconds=3600)
        
        image_link_base_256 = f"{CLOUDFLARE_BASE_URL}{base_url_256}"
        image_link_base_fullsize = f"{CLOUDFLARE_BASE_URL}{base_url_fullsize}"
    else:
        # Generate local tokens
        current_time = datetime.utcnow()
        new_expiration = current_time + timedelta(hours=MEDIA_TOKEN_EXPIRE_HOURS)

        token_256 = generate_img_token(base_url_256, new_expiration, current_user)
        token_fullsize = generate_img_token(base_url_fullsize, new_expiration, current_user)
        
        token_url_256 = f"{base_url_256}?token={token_256}"
        token_url_fullsize = f"{base_url_fullsize}?token={token_fullsize}"
        
        # Use provided scheme, host and port or extract them from request object
        if scheme is None or host is None:
            if request is not None:
                scheme = request.url.scheme
                host = request.url.hostname
                port = request.url.port
            else:
                raise ValueError("Cannot determine scheme and host without request or scheme and host parameters.")

        image_link_token_256 = f'{CLOUDFLARE_BASE_URL}{token_url_256}'
        image_link_token_fullsize = f'{CLOUDFLARE_BASE_URL}{token_url_fullsize}'

        image_link_base_256 = f'{CLOUDFLARE_BASE_URL}{base_url_256}'
        image_link_base_fullsize = f'{CLOUDFLARE_BASE_URL}{base_url_fullsize}'

    return image_link_base_256, image_link_token_256, image_link_base_fullsize, image_link_token_fullsize


def generate_img_token(string_to_use: str, expiration: datetime, current_user: User = Depends(get_current_user)) -> str:
    if not isinstance(string_to_use, str):
        raise TypeError(f"string_to_use must be a string, got {type(string_to_use)}")
    
    payload = {
        "exp": expiration,
        "username": current_user.username
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


async def get_or_generate_img_token(current_user: User = Depends(get_current_user)):
    user_id = current_user.id
    current_time = datetime.utcnow()
    new_expiration = current_time + timedelta(hours=MEDIA_TOKEN_EXPIRE_HOURS)

    if USE_REDIS:
        redis_key = f"img_token:{user_id}"
        token = redis_client.get(redis_key)

        if token:
            return token

        new_token = generate_img_token(f"new token for {user_id}", new_expiration, current_user)
        redis_client.setex(
            redis_key,
            timedelta(hours=MEDIA_TOKEN_EXPIRE_HOURS),
            new_token
        )
        return new_token
    else:
        cursor_mem = await conn_mem.cursor()
        await cursor_mem.execute("SELECT last_access, token FROM last_access WHERE user_id = ?", (user_id,))
        row = await cursor_mem.fetchone()

        if row:
            last_access, token = row
            last_access = datetime.strptime(last_access, '%Y-%m-%d %H:%M:%S')
            if current_time - last_access < timedelta(hours=MEDIA_TOKEN_EXPIRE_HOURS):
                return token

        new_token = generate_img_token(f"new token for {user_id}", new_expiration, current_user)
        await cursor_mem.execute(
            "INSERT OR REPLACE INTO last_access (user_id, last_access, token) VALUES (?, ?, ?)",
            (user_id, new_expiration.strftime('%Y-%m-%d %H:%M:%S'), new_token)
        )
        await conn_mem.commit()
        return new_token


def resize_image(image: PilImage.Image, size: int) -> PilImage.Image:
    """Resize an image to a square with the given side length."""
    if image.width != image.height:
        # Crop to square
        min_dimension = min(image.width, image.height)
        left = (image.width - min_dimension) / 2
        top = (image.height - min_dimension) / 2
        right = (image.width + min_dimension) / 2
        bottom = (image.height + min_dimension) / 2
        image = image.crop((left, top, right, bottom))
    
    return image.resize((size, size), PilImage.LANCZOS)


async def initialize_memory_db():
    global conn_mem
    if not USE_REDIS:
        conn_mem = await aiosqlite.connect(':memory:')
        cursor_mem = await conn_mem.cursor()
        await cursor_mem.execute('''
            CREATE TABLE last_access (
                user_id INTEGER PRIMARY KEY,
                last_access TIMESTAMP,
                token TEXT
            )
        ''')
        await conn_mem.commit()
        logger.info("In-memory SQLite database initialized")
    
    # Initialize cost system (non-blocking)
    try:
        await Cost.initialize()
    except Exception as e:
        logger.warning(f"Could not initialize Cost system during startup: {e}")
        logger.info("Cost system will use default values")


async def close_memory_db():
    global conn_mem
    if not USE_REDIS and conn_mem:
        await conn_mem.close()
        conn_mem = None
        logger.info("In-memory SQLite connection closed")