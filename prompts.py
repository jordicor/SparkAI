import io
import os
import orjson
import shutil
import logging
from PIL import Image as PilImage
from datetime import datetime, timedelta
from typing import Union, Optional, List, Dict
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi import FastAPI, APIRouter, Depends, File, Form, HTTPException, UploadFile, Request, status
from log_config import logger

# Imports from your own modules
from models import User
from auth import get_current_user
from database import get_db_connection
from save_images import generate_img_token, resize_image, get_or_generate_img_token
from common import (
    AVATAR_TOKEN_EXPIRE_HOURS,
    CLOUDFLARE_BASE_URL,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_UPLOAD_SIZE,
    consume_token,
    decrypt_api_key,
    generate_user_hash,
    get_llm_token_costs,
    get_template_context,
    get_user_api_key_mode,
    resolve_api_key_for_provider,
    sanitize_name,
    slugify,
    templates,
    users_directory,
)
from security_config import is_forbidden_prompt_name

router = APIRouter()

@router.get("/prompts", response_class=HTMLResponse)
async def list_prompts(request: Request, current_user: User = Depends(get_current_user)):
    if current_user is None:
        return templates.TemplateResponse("login.html", {"request": request})
        
    if not await current_user.is_admin and not await current_user.is_manager:
        raise HTTPException(status_code=403, detail="Access denied")
    
    async with get_db_connection(readonly=True) as conn:
        query = """
            SELECT p.id, p.name, p.description, p.image, p.created_at, p.public,
                   u_creator.username as created_by_username, v.name as voice_name,
                   COALESCE(u_owner.username, u_creator.username) as owner_username,
                   CASE WHEN pp_edit.user_id IS NOT NULL THEN 1 ELSE 0 END as can_edit,
                   CASE WHEN pp_owner.user_id IS NOT NULL THEN 1 ELSE 0 END as is_owner
            FROM Prompts p
            LEFT JOIN Voices v ON p.voice_id = v.id
            LEFT JOIN Users u_creator ON p.created_by_user_id = u_creator.id
            LEFT JOIN PROMPT_PERMISSIONS pp_owner_data ON p.id = pp_owner_data.prompt_id AND pp_owner_data.permission_level = 'owner'
            LEFT JOIN Users u_owner ON pp_owner_data.user_id = u_owner.id
            LEFT JOIN PROMPT_PERMISSIONS pp_edit ON p.id = pp_edit.prompt_id AND pp_edit.user_id = ? AND pp_edit.permission_level = 'edit'
            LEFT JOIN PROMPT_PERMISSIONS pp_owner ON p.id = pp_owner.prompt_id AND pp_owner.user_id = ? AND pp_owner.permission_level = 'owner'
        """
        params = [current_user.id, current_user.id]
        
        if await current_user.is_manager and not await current_user.is_admin:
            query += " WHERE pp_edit.user_id IS NOT NULL OR pp_owner.user_id IS NOT NULL"
        
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            prompts = []
            for row in rows:
                prompt = dict(row)
                prompt['created_at'] = datetime.strptime(prompt['created_at'], '%Y-%m-%d %H:%M:%S.%f')
                prompt['can_edit'] = bool(prompt['can_edit']) or bool(prompt['is_owner']) or await current_user.is_admin

                # Generate image URLs (thumbnail and fullsize)
                prompt_image_url = None
                prompt_image_fullsize_url = None
                if prompt['image']:
                    current_time = datetime.utcnow()
                    new_expiration = current_time + timedelta(hours=AVATAR_TOKEN_EXPIRE_HOURS)
                    # Thumbnail (32px)
                    image_base_url = f"{prompt['image']}_32.webp"
                    token = generate_img_token(image_base_url, new_expiration, current_user)
                    prompt_image_url = f"{CLOUDFLARE_BASE_URL}{image_base_url}?token={token}"
                    # Fullsize
                    image_fullsize_url = f"{prompt['image']}_fullsize.webp"
                    token_fullsize = generate_img_token(image_fullsize_url, new_expiration, current_user)
                    prompt_image_fullsize_url = f"{CLOUDFLARE_BASE_URL}{image_fullsize_url}?token={token_fullsize}"
                prompt['image_url'] = prompt_image_url
                prompt['image_fullsize_url'] = prompt_image_fullsize_url

                prompts.append(prompt)

        # Fetch categories for all prompts
        prompt_ids = [p['id'] for p in prompts]
        if prompt_ids:
            placeholders = ','.join('?' * len(prompt_ids))
            async with conn.execute(f"""
                SELECT pc.prompt_id, c.id, c.name, c.icon
                FROM PROMPT_CATEGORIES pc
                JOIN CATEGORIES c ON pc.category_id = c.id
                WHERE pc.prompt_id IN ({placeholders})
                ORDER BY c.display_order
            """, prompt_ids) as cursor:
                cat_rows = await cursor.fetchall()

            # Group categories by prompt_id
            prompt_categories = {}
            for cat_row in cat_rows:
                pid = cat_row[0]
                if pid not in prompt_categories:
                    prompt_categories[pid] = []
                prompt_categories[pid].append({
                    'id': cat_row[1],
                    'name': cat_row[2],
                    'icon': cat_row[3]
                })

            # Assign categories to each prompt
            for prompt in prompts:
                prompt['categories'] = prompt_categories.get(prompt['id'], [])
        
        context = await get_template_context(request, current_user)
        context["prompts"] = prompts
        return templates.TemplateResponse("prompts/prompt_list.html", context)

        
async def process_prompt_image_upload(
    prompt_id: int,
    file: UploadFile,
    prompt_info: Dict,
    current_user: User
):
    # Verify that current_user has permissions
    is_admin = await current_user.is_admin
    is_owner = prompt_info['created_by_user_id'] == current_user.id

    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            # Get the prompt owner
            await cursor.execute("""
                SELECT u.username 
                FROM Users u 
                JOIN PROMPT_PERMISSIONS pp ON u.id = pp.user_id 
                WHERE pp.prompt_id = ? AND pp.permission_level = 'owner'
            """, (prompt_id,))
            owner_result = await cursor.fetchone()
            
            if not owner_result:
                raise HTTPException(status_code=404, detail="Prompt owner not found")
            
            owner_username = owner_result[0]

            # Verify editing permissions
            await cursor.execute(
                "SELECT permission_level FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND user_id = ?",
                (prompt_id, current_user.id)
            )
            permission = await cursor.fetchone()
            is_editor = permission and permission[0] == 'edit'

    if not (is_admin or is_owner or is_editor):
        raise HTTPException(status_code=403, detail="Access denied")

    # Prepare the directory and filenames
    hash_prefix1, hash_prefix2, user_hash = generate_user_hash(owner_username)
    sanitized_prompt_name = sanitize_name(prompt_info['name'])
    padded_id = f"{prompt_id:07d}"
    prompt_dir = os.path.join(
        users_directory, 
        hash_prefix1, 
        hash_prefix2, 
        user_hash, 
        "prompts", 
        padded_id[:3], 
        f"{padded_id[3:]}_{sanitized_prompt_name}", 
        "static", 
        "img"
    )

    # Create directory if it does not exist
    os.makedirs(prompt_dir, exist_ok=True)

    try:
        # Process the new image
        content = await file.read()

        # Security: Check file size limit
        if len(content) > MAX_IMAGE_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"Image too large. Maximum size is {MAX_IMAGE_UPLOAD_SIZE // (1024*1024)}MB")

        image = PilImage.open(io.BytesIO(content))

        # Security: Check for decompression bombs (excessive pixel count)
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail=f"Image dimensions too large. Maximum is {MAX_IMAGE_PIXELS:,} pixels")

        sizes = [32, 64, 128, 'fullsize']
        ext = 'webp'

        # Create the new image versions
        for size in sizes:
            if size == 'fullsize':
                resized_image = image
                filename = f"{prompt_id}_{sanitized_prompt_name}_fullsize.{ext}"
            else:
                resized_image = resize_image(image, size)
                filename = f"{prompt_id}_{sanitized_prompt_name}_{size}.{ext}"

            file_path = os.path.join(prompt_dir, filename)
            resized_image.save(file_path, ext.upper())

        # Build the base_image_url (without timestamp)
        base_image_url = f"users/{hash_prefix1}/{hash_prefix2}/{user_hash}/prompts/{padded_id[:3]}/{padded_id[3:]}_{sanitized_prompt_name}/static/img/{prompt_id}_{sanitized_prompt_name}"

        # Update the database
        async with get_db_connection() as conn:
            await conn.execute("UPDATE Prompts SET image = ? WHERE id = ?", (base_image_url, prompt_id))
            await conn.commit()

        return base_image_url

    except Exception as e:
        logger.error(f"Error processing image for prompt {prompt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/prompts/new", response_class=HTMLResponse)
async def create_prompt(request: Request, current_user: User = Depends(get_current_user)):
    if current_user is None:
        return templates.TemplateResponse("login.html", {"request": request})
        
    if not await current_user.is_admin and not await current_user.is_manager:
        raise HTTPException(status_code=403, detail="Access denied")
    async with get_db_connection(readonly=True) as conn:
        async with conn.execute("SELECT voice_code, name FROM Voices") as cursor:
            voices = await cursor.fetchall()
        context = await get_template_context(request, current_user)
        context["voices"] = voices
        return templates.TemplateResponse("prompts/create_prompt.html", context)

@router.post("/prompts/new")
async def create_prompt_post(
    request: Request,
    current_user: User = Depends(get_current_user),
    name: str = Form(...),
    prompt: str = Form(...),
    description: str = Form(...),
    sample_voice_id: str = Form(...),
    public: bool = Form(False),
    image: UploadFile = File(None),
    category_ids: str = Form(""),
    # Pricing fields
    is_paid: int = Form(0),
    markup_per_mtokens: float = Form(0.0),
    llm_mode: str = Form("any"),
    forced_llm_id: Optional[int] = Form(None),
    hide_llm_name: bool = Form(False),
    disable_web_search: bool = Form(False),
    enable_moderation: bool = Form(False),
    watchdog_config: Optional[str] = Form(None)
):
    if current_user is None:
        return templates.TemplateResponse("login.html", {"request": request})

    if not await current_user.is_admin and not await current_user.is_manager:
        raise HTTPException(status_code=403, detail="Access denied")

    # Validate prompt name is not forbidden (security)
    if is_forbidden_prompt_name(name) or is_forbidden_prompt_name(slugify(name)):
        raise HTTPException(status_code=400, detail="This name is not available. Please choose a different name.")

    # Parse and validate watchdog_config
    watchdog_config_json = None
    if watchdog_config and watchdog_config.strip():
        try:
            parsed_wd = orjson.loads(watchdog_config)
            sanitized_wd = validate_watchdog_config(parsed_wd)
            # Async FK check for llm_id on both sub-configs
            for sub_key in ("pre_watchdog", "post_watchdog"):
                sub_cfg = sanitized_wd.get(sub_key, {})
                if sub_cfg.get("enabled") and sub_cfg.get("llm_id") is not None:
                    async with get_db_connection(readonly=True) as conn_check:
                        cursor_check = await conn_check.execute("SELECT id FROM LLM WHERE id = ?", (sub_cfg["llm_id"],))
                        if not await cursor_check.fetchone():
                            raise ValueError(f"LLM with id {sub_cfg['llm_id']} does not exist ({sub_key})")
            watchdog_config_json = orjson.dumps(sanitized_wd).decode("utf-8")
        except orjson.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in watchdog configuration")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Watchdog config error: {e}")

    # Parse category_ids
    parsed_category_ids = []
    if category_ids:
        try:
            parsed_category_ids = orjson.loads(category_ids)
            if not isinstance(parsed_category_ids, list):
                parsed_category_ids = []
        except orjson.JSONDecodeError:
            parsed_category_ids = []

    # Validate: public prompts require at least one category
    if public and not parsed_category_ids:
        raise HTTPException(status_code=400, detail="Public prompts require at least one category")

    prompt_id = None
    async with get_db_connection() as conn:
        try:
            await conn.execute("BEGIN TRANSACTION")

            # Find the voice_id from the sample_voice_id
            cursor = await conn.execute(
                "SELECT id FROM Voices WHERE voice_code = ?",
                (sample_voice_id,)
            )
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Voice not found")
            voice_id = row['id']

            # Process pricing fields
            is_paid_bool = bool(is_paid)
            actual_forced_llm_id = forced_llm_id if llm_mode == "forced" and forced_llm_id else None
            actual_hide_llm_name = hide_llm_name if llm_mode == "forced" else False
            actual_markup = markup_per_mtokens if is_paid_bool else 0.0

            # Insert the new prompt with the found voice_id and pricing fields
            cursor = await conn.execute(
                """INSERT INTO Prompts (name, prompt, description, voice_id, created_by_user_id, created_at, public,
                   is_paid, markup_per_mtokens, forced_llm_id, hide_llm_name, disable_web_search, enable_moderation,
                   watchdog_config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, prompt, description, voice_id, current_user.id, datetime.utcnow(), public,
                 is_paid_bool, actual_markup, actual_forced_llm_id, actual_hide_llm_name, disable_web_search, enable_moderation,
                 watchdog_config_json)
            )
            prompt_id = cursor.lastrowid

            await conn.execute(
                "INSERT INTO PROMPT_PERMISSIONS (prompt_id, user_id, permission_level) VALUES (?, ?, ?)",
                (prompt_id, current_user.id, 'owner')
            )

            # Insert categories
            for cat_id in parsed_category_ids:
                await conn.execute(
                    "INSERT INTO PROMPT_CATEGORIES (prompt_id, category_id) VALUES (?, ?)",
                    (prompt_id, cat_id)
                )

            await conn.execute("COMMIT")
        except Exception as e:
            await conn.execute("ROLLBACK")
            raise HTTPException(status_code=500, detail=f"Error creating prompt: {str(e)}")

    if prompt_id:
        prompt_info = await get_prompt_info(prompt_id)
        create_prompt_directory(current_user.username, prompt_id, prompt_info['name'])

        if image and image.filename:
            await process_prompt_image_upload(prompt_id, image, prompt_info, current_user)

    return RedirectResponse(url="/prompts", status_code=303)


@router.get("/prompts/edit/{prompt_id}", response_class=HTMLResponse)
async def edit_prompt(request: Request, prompt_id: int, current_user: User = Depends(get_current_user)):
    if current_user is None:
        return templates.TemplateResponse("login.html", {"request": request})
    
    async with get_db_connection(readonly=True) as conn:
        # Get prompt information including pricing fields
        async with conn.execute("""SELECT name, prompt, description, voice_id, image, created_by_user_id, public,
                                          is_paid, markup_per_mtokens, forced_llm_id, hide_llm_name, disable_web_search,
                                          enable_moderation, watchdog_config
                                   FROM Prompts WHERE id = ?""", (prompt_id,)) as cursor:
            prompt = await cursor.fetchone()
        
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        # Verify permissions
        async with conn.execute("SELECT role_name FROM USER_ROLES WHERE id = ?", (current_user.role_id,)) as cursor:
            role_result = await cursor.fetchone()
            role_name = role_result[0] if role_result else None

        is_admin = role_name == 'admin'
        is_manager = role_name == 'manager'
        
        # Get the current owner
        async with conn.execute("SELECT user_id FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND permission_level = 'owner'", (prompt_id,)) as cursor:
            owner_result = await cursor.fetchone()
            current_owner_id = owner_result[0] if owner_result else None
        
        is_owner = current_owner_id == current_user.id
        
        # Check if the user has edit permission
        async with conn.execute("SELECT 1 FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND user_id = ? AND permission_level = 'edit'", (prompt_id, current_user.id)) as cursor:
            is_editor = await cursor.fetchone() is not None
        
        if not (is_admin or is_owner or is_editor):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get the voice_code corresponding to voice_id
        async with conn.execute("SELECT voice_code FROM Voices WHERE id = ?", (prompt[3],)) as cursor:
            voice_code_result = await cursor.fetchone()
            voice_code = voice_code_result[0] if voice_code_result else None
        
        # Get voices
        async with conn.execute("SELECT id, name FROM Voices") as cursor:
            voices = await cursor.fetchall()
        
        # Get users who are admins or managers
        async with conn.execute("""
            SELECT u.id, u.username 
            FROM USERS u
            JOIN USER_ROLES ur ON u.role_id = ur.id
            WHERE ur.role_name IN ('admin', 'manager')
            ORDER BY u.id
        """) as cursor:
            users = await cursor.fetchall()
        
        # Get editors
        async with conn.execute("""
            SELECT u.id, u.username 
            FROM Users u 
            JOIN PROMPT_PERMISSIONS pp ON u.id = pp.user_id 
            JOIN USER_ROLES ur ON u.role_id = ur.id
            WHERE pp.prompt_id = ? AND pp.permission_level = 'edit'
            AND ur.role_name IN ('admin', 'manager')
        """, (prompt_id,)) as cursor:
            editors = await cursor.fetchall()
        
        editor_ids = [editor[0] for editor in editors]

    # Generate prompt image URL
    prompt_image_url = None
    if prompt[4]:  # prompt[4] is the 'image' column in the Prompts table
        current_time = datetime.utcnow()
        new_expiration = current_time + timedelta(hours=AVATAR_TOKEN_EXPIRE_HOURS)
        image_base_url = f"{prompt[4]}_128.webp"
        token = generate_img_token(image_base_url, new_expiration, current_user)
        prompt_image_url = f"{CLOUDFLARE_BASE_URL}{image_base_url}?token={token}"

    context = await get_template_context(request, current_user)
    context.update({
        "prompt_id": prompt_id,
        "prompt_name": prompt[0],
        "prompt_text": prompt[1],
        "description": prompt[2],
        "voice_code": voice_code,
        "image_url": prompt_image_url,
        "created_by_user_id": prompt[5],
        "current_owner_id": current_owner_id,
        "is_public": prompt[6],
        "voices": voices,
        "users": users,
        "is_owner": is_owner,
        "is_editor": is_editor,
        "editors": editors,
        "editor_ids": editor_ids,
        # Pricing fields
        "is_paid": prompt[7] if prompt[7] else False,
        "markup_per_mtokens": prompt[8] if prompt[8] else 0.0,
        "forced_llm_id": prompt[9],
        "hide_llm_name": prompt[10] if prompt[10] else False,
        # Web search control
        "disable_web_search": prompt[11] if prompt[11] else False,
        # Content moderation
        "enable_moderation": prompt[12] if prompt[12] else False,
        # Watchdog config
        "watchdog_config": _parse_watchdog_config_for_template(prompt[13]),
    })
    return templates.TemplateResponse("prompts/edit_prompt.html", context)


@router.post("/prompts/update/{prompt_id}")
async def update_prompt(
    request: Request,
    prompt_id: int,
    current_user: User = Depends(get_current_user),
    name: str = Form(...),
    prompt: str = Form(...),
    description: str = Form(...),
    sample_voice_id: str = Form(...),
    public: bool = Form(False),
    image: Optional[UploadFile] = File(None),
    editor_ids: Optional[str] = Form(None),
    new_owner_id: Optional[int] = Form(None),
    category_ids: Optional[str] = Form(""),
    # Pricing fields
    is_paid: int = Form(0),
    markup_per_mtokens: float = Form(0.0),
    llm_mode: str = Form("any"),
    forced_llm_id: Optional[int] = Form(None),
    hide_llm_name: bool = Form(False),
    disable_web_search: bool = Form(False),
    enable_moderation: bool = Form(False),
    watchdog_config: Optional[str] = Form(None)
):
    if current_user is None:
        return templates.TemplateResponse("login.html", {"request": request})

    prompt_info = await get_prompt_info(prompt_id)

    # Validate prompt name is not forbidden (security)
    # Only check if name is being changed
    if prompt_info and name != prompt_info.get('name', ''):
        if is_forbidden_prompt_name(name) or is_forbidden_prompt_name(slugify(name)):
            raise HTTPException(status_code=400, detail="This name is not available. Please choose a different name.")

    # Parse and validate watchdog_config
    watchdog_config_json = None
    if watchdog_config and watchdog_config.strip():
        try:
            parsed_wd = orjson.loads(watchdog_config)
            sanitized_wd = validate_watchdog_config(parsed_wd)
            # Async FK check for llm_id on both sub-configs
            for sub_key in ("pre_watchdog", "post_watchdog"):
                sub_cfg = sanitized_wd.get(sub_key, {})
                if sub_cfg.get("enabled") and sub_cfg.get("llm_id") is not None:
                    async with get_db_connection(readonly=True) as conn_check:
                        cursor_check = await conn_check.execute("SELECT id FROM LLM WHERE id = ?", (sub_cfg["llm_id"],))
                        if not await cursor_check.fetchone():
                            raise ValueError(f"LLM with id {sub_cfg['llm_id']} does not exist ({sub_key})")
            watchdog_config_json = orjson.dumps(sanitized_wd).decode("utf-8")
        except orjson.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in watchdog configuration")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Watchdog config error: {e}")

    # Parse category_ids
    parsed_category_ids = []
    if category_ids:
        try:
            parsed_category_ids = orjson.loads(category_ids)
            if not isinstance(parsed_category_ids, list):
                parsed_category_ids = []
        except orjson.JSONDecodeError:
            parsed_category_ids = []

    # Validate: public prompts require at least one category
    if public and not parsed_category_ids:
        raise HTTPException(status_code=400, detail="Public prompts require at least one category")

    async with get_db_connection() as conn:
        # Verify permissions
        async with conn.execute("SELECT role_name FROM USER_ROLES WHERE id = ?", (current_user.role_id,)) as cursor:
            role_result = await cursor.fetchone()
            is_admin = role_result[0] == 'admin' if role_result else False

        async with conn.execute("SELECT user_id FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND permission_level = 'owner'", (prompt_id,)) as cursor:
            owner_result = await cursor.fetchone()
            current_owner_id = owner_result[0] if owner_result else None

        is_owner = current_owner_id == current_user.id

        if not (is_admin or is_owner):
            raise HTTPException(status_code=403, detail="Access denied")

        # Get public_id for cache invalidation
        async with conn.execute("SELECT public_id FROM PROMPTS WHERE id = ?", (prompt_id,)) as cursor:
            public_id_result = await cursor.fetchone()
            prompt_public_id = public_id_result[0] if public_id_result else None

        editor_ids_list = []
        if editor_ids:
            try:
                editor_ids_list = orjson.loads(editor_ids)
                if not isinstance(editor_ids_list, list):
                    raise ValueError("editor_ids should be a list")
            except orjson.JSONDecodeError:
                editor_ids_list = [int(id.strip()) for id in editor_ids.split(',') if id.strip()]

    # Process the image
    if image and image.filename:
        await process_prompt_image_upload(prompt_id, image, prompt_info, current_user)

    async with get_db_connection() as conn:
        async with conn.cursor() as cursor:
            # Find the voice_id from the sample_voice_id
            await cursor.execute(
                "SELECT id FROM Voices WHERE voice_code = ?",
                (sample_voice_id,)
            )
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Voice not found")
            voice_id = row['id']

            # Process pricing fields
            is_paid_bool = bool(is_paid)
            actual_forced_llm_id = forced_llm_id if llm_mode == "forced" and forced_llm_id else None
            actual_hide_llm_name = hide_llm_name if llm_mode == "forced" else False
            actual_markup = markup_per_mtokens if is_paid_bool else 0.0

            # Update the prompt information including pricing fields
            await cursor.execute(
                """UPDATE Prompts SET name = ?, prompt = ?, description = ?, voice_id = ?, public = ?,
                   is_paid = ?, markup_per_mtokens = ?, forced_llm_id = ?, hide_llm_name = ?, disable_web_search = ?,
                   enable_moderation = ?, watchdog_config = ?
                   WHERE id = ?""",
                (name, prompt, description, voice_id, public,
                 is_paid_bool, actual_markup, actual_forced_llm_id, actual_hide_llm_name, disable_web_search,
                 enable_moderation, watchdog_config_json, prompt_id)
            )

            # Update or create the owner
            if new_owner_id:
                if current_owner_id:
                    await cursor.execute(
                        "UPDATE PROMPT_PERMISSIONS SET user_id = ? WHERE prompt_id = ? AND permission_level = 'owner'",
                        (new_owner_id, prompt_id)
                    )
                else:
                    await cursor.execute(
                        "INSERT INTO PROMPT_PERMISSIONS (prompt_id, user_id, permission_level) VALUES (?, ?, 'owner')",
                        (prompt_id, new_owner_id)
                    )
            elif not current_owner_id and is_admin:
                # If there is no current owner and the user is admin, assign the admin as owner
                await cursor.execute(
                    "INSERT INTO PROMPT_PERMISSIONS (prompt_id, user_id, permission_level) VALUES (?, ?, 'owner')",
                    (prompt_id, current_user.id)
                )

            # Update the editors
            await cursor.execute("DELETE FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND permission_level = 'edit'", (prompt_id,))
            for editor_id in editor_ids_list:
                await cursor.execute(
                    "INSERT INTO PROMPT_PERMISSIONS (prompt_id, user_id, permission_level) VALUES (?, ?, 'edit')",
                    (prompt_id, editor_id)
                )

            # Update categories - delete existing and insert new
            await cursor.execute("DELETE FROM PROMPT_CATEGORIES WHERE prompt_id = ?", (prompt_id,))
            for cat_id in parsed_category_ids:
                await cursor.execute(
                    "INSERT INTO PROMPT_CATEGORIES (prompt_id, category_id) VALUES (?, ?)",
                    (prompt_id, cat_id)
                )

        await conn.commit()

    # Invalidate landing cache if prompt has a public_id
    if prompt_public_id:
        from app import invalidate_landing_cache
        invalidate_landing_cache(prompt_public_id)

    return RedirectResponse(url=f"/prompts/edit/{prompt_id}", status_code=303)


@router.delete("/prompts/delete/{prompt_id}")
async def delete_prompt(prompt_id: int, current_user: User = Depends(get_current_user)):
    if current_user is None:
        return templates.TemplateResponse("login.html", {"request": request})
    
    async with get_db_connection() as conn:
        # Verify if the user is admin or owner
        is_admin = await current_user.is_admin
        
        # First we get the prompt information and its owner
        async with conn.execute("""
            SELECT p.*, u.username as owner_username
            FROM Prompts p
            JOIN PROMPT_PERMISSIONS pp ON p.id = pp.prompt_id
            JOIN Users u ON pp.user_id = u.id
            WHERE p.id = ? AND pp.permission_level = 'owner'
        """, (prompt_id,)) as cursor:
            prompt_info = await cursor.fetchone()

        if not prompt_info:
            raise HTTPException(status_code=404, detail="Prompt not found")

        async with conn.execute("SELECT user_id FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND permission_level = 'owner'", (prompt_id,)) as cursor:
            owner_result = await cursor.fetchone()
            is_owner = owner_result and owner_result[0] == current_user.id

        if not (is_admin or is_owner):
            raise HTTPException(status_code=403, detail="Access denied")

        # Save public_id for cache invalidation (before deleting)
        prompt_public_id = prompt_info['public_id'] if 'public_id' in prompt_info.keys() else None

        try:
            # Delete the prompt from the database
            cursor = await conn.execute('DELETE FROM Prompts WHERE id = ? RETURNING id', (prompt_id,))
            deleted = await cursor.fetchone()
            
            if not deleted:
                raise HTTPException(status_code=404, detail="Prompt not found")
            
            # Delete associated permissions
            await conn.execute('DELETE FROM PROMPT_PERMISSIONS WHERE prompt_id = ?', (prompt_id,))
            
            # Delete the prompt folder and its contents
            hash_prefix1, hash_prefix2, user_hash = generate_user_hash(prompt_info['owner_username'])
            sanitized_prompt_name = sanitize_name(prompt_info['name'])
            padded_id = f"{prompt_id:07d}"
            prompt_dir = os.path.join(
                users_directory,
                hash_prefix1,
                hash_prefix2,
                user_hash,
                "prompts",
                padded_id[:3],
                f"{padded_id[3:]}_{sanitized_prompt_name}"
            )
            
            if os.path.exists(prompt_dir):
                try:
                    shutil.rmtree(prompt_dir)
                except Exception as e:
                    logger.error(f"Error deleting prompt directory {prompt_dir}: {e}")
            
            await conn.commit()

            # Invalidate landing cache
            if prompt_public_id:
                from app import invalidate_landing_cache
                invalidate_landing_cache(prompt_public_id)

            return JSONResponse(content={"success": True}, status_code=200)

        except Exception as e:
            await conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/delete-batch")
async def delete_prompts_batch(request: Request, current_user: User = Depends(get_current_user)):
    """Delete multiple prompts at once"""
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    if not await current_user.is_admin and not await current_user.is_manager:
        raise HTTPException(status_code=403, detail="Access denied")

    form_data = await request.form()
    selected_prompts = form_data.getlist("selected_prompts")

    if not selected_prompts:
        return RedirectResponse(url="/prompts", status_code=303)

    deleted_count = 0
    errors = []

    for prompt_id_str in selected_prompts:
        try:
            prompt_id = int(prompt_id_str)

            async with get_db_connection() as conn:
                is_admin = await current_user.is_admin

                # Get prompt information and owner
                async with conn.execute("""
                    SELECT p.*, u.username as owner_username
                    FROM Prompts p
                    JOIN PROMPT_PERMISSIONS pp ON p.id = pp.prompt_id
                    JOIN Users u ON pp.user_id = u.id
                    WHERE p.id = ? AND pp.permission_level = 'owner'
                """, (prompt_id,)) as cursor:
                    prompt_info = await cursor.fetchone()

                if not prompt_info:
                    errors.append(f"Prompt {prompt_id} not found")
                    continue

                # Check ownership
                async with conn.execute(
                    "SELECT user_id FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND permission_level = 'owner'",
                    (prompt_id,)
                ) as cursor:
                    owner_result = await cursor.fetchone()
                    is_owner = owner_result and owner_result[0] == current_user.id

                # Check edit permission for managers
                async with conn.execute(
                    "SELECT permission_level FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND user_id = ? AND permission_level IN ('owner', 'edit')",
                    (prompt_id, current_user.id)
                ) as cursor:
                    has_permission = await cursor.fetchone() is not None

                if not (is_admin or is_owner or has_permission):
                    errors.append(f"No permission to delete prompt {prompt_id}")
                    continue

                # Save public_id for cache invalidation (before deleting)
                prompt_public_id = prompt_info['public_id'] if 'public_id' in prompt_info.keys() else None

                # Delete prompt
                await conn.execute('DELETE FROM Prompts WHERE id = ?', (prompt_id,))
                await conn.execute('DELETE FROM PROMPT_PERMISSIONS WHERE prompt_id = ?', (prompt_id,))

                # Delete prompt directory
                hash_prefix1, hash_prefix2, user_hash = generate_user_hash(prompt_info['owner_username'])
                sanitized_prompt_name = sanitize_name(prompt_info['name'])
                padded_id = f"{prompt_id:07d}"
                prompt_dir = os.path.join(
                    users_directory,
                    hash_prefix1,
                    hash_prefix2,
                    user_hash,
                    "prompts",
                    padded_id[:3],
                    f"{padded_id[3:]}_{sanitized_prompt_name}"
                )

                if os.path.exists(prompt_dir):
                    try:
                        shutil.rmtree(prompt_dir)
                    except Exception as e:
                        logger.error(f"Error deleting prompt directory {prompt_dir}: {e}")

                await conn.commit()

                # Invalidate landing cache
                if prompt_public_id:
                    from app import invalidate_landing_cache
                    invalidate_landing_cache(prompt_public_id)

                deleted_count += 1

        except ValueError:
            errors.append(f"Invalid prompt ID: {prompt_id_str}")
        except Exception as e:
            errors.append(f"Error deleting prompt {prompt_id_str}: {str(e)}")

    return RedirectResponse(url="/prompts", status_code=303)


@router.post("/api/delete-prompt-image/{prompt_id}")
async def delete_prompt_image(prompt_id: int, current_user: User = Depends(get_current_user)):
    if current_user is None:
        return JSONResponse(content={"success": False, "message": "User not authenticated"}, status_code=401)

    async with get_db_connection() as conn:
        # Get prompt information and its owner
        async with conn.execute("""
            SELECT p.*, u.username as owner_username
            FROM Prompts p
            JOIN PROMPT_PERMISSIONS pp ON p.id = pp.prompt_id
            JOIN Users u ON pp.user_id = u.id
            WHERE p.id = ? AND pp.permission_level = 'owner'
        """, (prompt_id,)) as cursor:
            prompt_info = await cursor.fetchone()

        if not prompt_info:
            return JSONResponse(content={"success": False, "message": "Prompt not found"}, status_code=404)

        # Verify permissions
        is_admin = await current_user.is_admin
        is_owner = prompt_info['created_by_user_id'] == current_user.id
        
        async with conn.execute(
            "SELECT permission_level FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND user_id = ? AND permission_level IN ('owner', 'edit')", 
            (prompt_id, current_user.id)
        ) as cursor:
            has_permission = await cursor.fetchone() is not None

        if not (is_admin or is_owner or has_permission):
            return JSONResponse(content={"success": False, "message": "Access denied"}, status_code=403)

        # Use the owner's username for the directory structure
        hash_prefix1, hash_prefix2, user_hash = generate_user_hash(prompt_info['owner_username'])
        sanitized_prompt_name = sanitize_name(prompt_info['name'])
        padded_id = f"{prompt_id:07d}"
        prompt_dir = os.path.join(users_directory, hash_prefix1, hash_prefix2, user_hash, "prompts", padded_id[:3], f"{padded_id[3:]}_{sanitized_prompt_name}", "static", "img")

        # Delete the image files
        sizes = [32, 64, 128, 'fullsize']
        for size in sizes:
            if size == 'fullsize':
                filename = f"{prompt_id}_{sanitized_prompt_name}_fullsize.webp"
            else:
                filename = f"{prompt_id}_{sanitized_prompt_name}_{size}.webp"
            file_path = os.path.join(prompt_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete image reference from the database
        await conn.execute("UPDATE Prompts SET image = NULL WHERE id = ?", (prompt_id,))
        await conn.commit()

    return JSONResponse(content={"success": True, "message": "Prompt image deleted successfully"}, status_code=200)
        
async def get_manager_accessible_prompts(manager_id: int):
    """Get all prompts a manager can assign to users (including public prompts).

    Used when assigning prompts to users - includes public prompts because
    users can be assigned to use any public prompt.
    """
    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            query = """
            SELECT DISTINCT p.id
            FROM PROMPTS p
            LEFT JOIN PROMPT_PERMISSIONS pp ON p.id = pp.prompt_id
            WHERE p.created_by_user_id = ?
               OR pp.user_id = ?
               OR p.public = TRUE
            """
            await cursor.execute(query, (manager_id, manager_id))
            prompts = await cursor.fetchall()
    return [prompt[0] for prompt in prompts]


async def get_manager_owned_prompts(manager_id: int):
    """Get only prompt IDs that a manager can manage.

    Used in configuration panels where managers should only see/edit their own
    prompts, NOT public prompts from other users that they cannot modify.

    Permission logic (consistent with can_manage_prompt):
    - Has 'owner' or 'edit' permission in PROMPT_PERMISSIONS, OR
    - Is the original creator AND no explicit owner has been assigned

    Note: If ownership is transferred (someone else has 'owner' permission),
    the original creator loses management rights.
    """
    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            query = """
            SELECT DISTINCT p.id
            FROM PROMPTS p
            LEFT JOIN PROMPT_PERMISSIONS pp ON p.id = pp.prompt_id AND pp.user_id = ?
            WHERE
                -- Has explicit owner or edit permission
                pp.permission_level IN ('owner', 'edit')
                OR (
                    -- Is creator AND no explicit owner exists for this prompt
                    p.created_by_user_id = ?
                    AND NOT EXISTS (
                        SELECT 1 FROM PROMPT_PERMISSIONS pp2
                        WHERE pp2.prompt_id = p.id AND pp2.permission_level = 'owner'
                    )
                )
            """
            await cursor.execute(query, (manager_id, manager_id))
            prompts = await cursor.fetchall()
    return [prompt[0] for prompt in prompts]


async def can_manage_prompt(user_id: int, prompt_id: int, is_admin: bool = False) -> bool:
    """
    Check if a user can manage a prompt (configure landing page, edit settings, etc).

    Permission logic:
    1. Admin -> always allowed
    2. Has 'owner' permission in PROMPT_PERMISSIONS -> allowed
    3. Has 'edit' permission in PROMPT_PERMISSIONS -> allowed
    4. No explicit 'owner' exists AND user is created_by_user_id -> allowed (default ownership)

    Note: Once an 'owner' is assigned via PROMPT_PERMISSIONS, the original creator
    loses management rights (unless they also have explicit permissions).
    This allows prompt ownership transfer.
    """
    if is_admin:
        return True

    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            # Check if user has explicit owner or edit permission
            await cursor.execute('''
                SELECT permission_level FROM PROMPT_PERMISSIONS
                WHERE prompt_id = ? AND user_id = ? AND permission_level IN ('owner', 'edit')
            ''', (prompt_id, user_id))
            user_permission = await cursor.fetchone()

            if user_permission:
                return True

            # Check if there's any explicit owner assigned
            await cursor.execute('''
                SELECT 1 FROM PROMPT_PERMISSIONS
                WHERE prompt_id = ? AND permission_level = 'owner'
                LIMIT 1
            ''', (prompt_id,))
            has_explicit_owner = await cursor.fetchone()

            if has_explicit_owner:
                # An owner exists and it's not the current user (already checked above)
                return False

            # No explicit owner - check if user is the original creator
            await cursor.execute('''
                SELECT 1 FROM PROMPTS WHERE id = ? AND created_by_user_id = ?
            ''', (prompt_id, user_id))
            is_creator = await cursor.fetchone()

            return is_creator is not None


async def get_manageable_prompts(user_id: int, is_admin: bool = False) -> list:
    """
    Get list of prompts that a user can manage (configure landing, edit, etc).

    Uses the same logic as can_manage_prompt():
    - Admin: all prompts
    - User with 'owner' or 'edit' permission
    - Original creator IF no explicit owner is assigned

    Returns list of dicts with prompt info for dropdown display.
    """
    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            if is_admin:
                await cursor.execute('''
                    SELECT p.id, p.name, u.username as created_by_username, p.public_id,
                           CASE WHEN pcd.is_active = 1 AND pcd.verification_status = 1
                                THEN pcd.custom_domain ELSE NULL END as custom_domain
                    FROM PROMPTS p
                    JOIN USERS u ON p.created_by_user_id = u.id
                    LEFT JOIN PROMPT_CUSTOM_DOMAINS pcd ON p.id = pcd.prompt_id
                    ORDER BY p.name
                ''')
            else:
                # Get prompts where:
                # 1. User has owner/edit permission, OR
                # 2. User is creator AND no explicit owner exists
                await cursor.execute('''
                    SELECT DISTINCT p.id, p.name, u.username as created_by_username, p.public_id,
                           CASE WHEN pcd.is_active = 1 AND pcd.verification_status = 1
                                THEN pcd.custom_domain ELSE NULL END as custom_domain
                    FROM PROMPTS p
                    JOIN USERS u ON p.created_by_user_id = u.id
                    LEFT JOIN PROMPT_CUSTOM_DOMAINS pcd ON p.id = pcd.prompt_id
                    LEFT JOIN PROMPT_PERMISSIONS pp ON p.id = pp.prompt_id AND pp.user_id = ?
                    WHERE
                        -- Has explicit owner or edit permission
                        pp.permission_level IN ('owner', 'edit')
                        OR (
                            -- Is creator AND no explicit owner exists for this prompt
                            p.created_by_user_id = ?
                            AND NOT EXISTS (
                                SELECT 1 FROM PROMPT_PERMISSIONS pp2
                                WHERE pp2.prompt_id = p.id AND pp2.permission_level = 'owner'
                            )
                        )
                    ORDER BY p.name
                ''', (user_id, user_id))

            rows = await cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'text': row[1],  # For compatibility with existing templates
                    'created_by_username': row[2],
                    'public_id': row[3],
                    'custom_domain': row[4]
                }
                for row in rows
            ]


# Function to get the user directory path
def get_user_directory(username: str) -> str:
    hash_prefix1, hash_prefix2, user_hash = generate_user_hash(username)
    return os.path.join(users_directory, hash_prefix1, hash_prefix2, user_hash)

def get_user_prompts_directory(username: str) -> str:
    user_dir = get_user_directory(username)
    return os.path.join(user_dir, "prompts")


# Function to create the prompt directory
def create_prompt_directory(username: str, prompt_id: Union[int, str], prompt_name: str) -> str:
    user_dir = get_user_directory(username)
    sanitized_name = sanitize_name(prompt_name)
    padded_id = f"{int(prompt_id):07d}"
    prompt_dir = os.path.join(user_dir, "prompts", padded_id[:3], f"{padded_id[3:]}_{sanitized_name}")
    
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)
    
    return prompt_dir

# Function to get prompt information
async def get_prompt_info(prompt_id: int) -> dict:
    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT p.name, p.created_by_user_id, u.username 
                FROM PROMPTS p
                JOIN USERS u ON p.created_by_user_id = u.id
                WHERE p.id = ?
            """, (prompt_id,))
            result = await cursor.fetchone()
            if result:
                return {
                    "name": result[0],
                    "created_by_user_id": result[1],
                    "created_by_username": result[2]
                }
            else:
                raise HTTPException(status_code=404, detail="Prompt not found")

# Function to get the prompt path
def get_prompt_path(prompt_id: int, prompt_info: dict) -> str:
    return create_prompt_directory(prompt_info['created_by_username'], prompt_id, prompt_info['name'])

# Function to get the prompt templates directory
def get_prompt_templates_dir(prompt_id: int, prompt_info: dict) -> str:
    prompt_dir = get_prompt_path(prompt_id, prompt_info)
    return os.path.join(prompt_dir, "templates")

# Function to get the prompt components directory
def get_prompt_components_dir(prompt_id: int, prompt_info: dict) -> str:
    templates_dir = get_prompt_templates_dir(prompt_id, prompt_info)
    return os.path.join(templates_dir, "components")


# =============================================================================
# Landing Registration Configuration
# =============================================================================

# Default configuration values for landing page registrations
DEFAULT_LANDING_REGISTRATION_CONFIG = {
    "default_llm_id": None,           # null = use prompt's forced_llm or system default (LLM id=1)
    "public_prompts_access": True,    # true = can browse public prompts, false = captive
    "allow_file_upload": False,
    "allow_image_generation": False,
    "initial_balance": 0.0,
    "billing_mode": "user_pays",      # "user_pays" | "manager_pays"
    "billing_limit": None,            # Only used when billing_mode = "manager_pays"
    "billing_limit_action": "block",  # "block" | "notify" | "auto_refill"
    "billing_auto_refill_amount": 10.0,  # Amount to increase limit by when auto_refill triggers
    "billing_max_limit": None,        # Maximum limit cap for auto_refill (null = unlimited)
    "category_access": None           # JSON array of category IDs or null (all)
}


async def get_landing_registration_config(prompt_id: int) -> dict:
    """
    Get the landing registration configuration for a prompt.
    Returns merged config with defaults for any missing keys.
    """
    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT landing_registration_config, forced_llm_id FROM PROMPTS WHERE id = ?",
                (prompt_id,)
            )
            result = await cursor.fetchone()

            if not result:
                raise HTTPException(status_code=404, detail="Prompt not found")

            config_json = result[0]
            forced_llm_id = result[1]

            # Start with defaults
            config = DEFAULT_LANDING_REGISTRATION_CONFIG.copy()

            # Override with stored config if exists
            if config_json:
                try:
                    stored_config = orjson.loads(config_json)
                    config.update(stored_config)
                except orjson.JSONDecodeError:
                    logger.warning(f"Invalid JSON in landing_registration_config for prompt {prompt_id}")

            # Add prompt's forced_llm_id for reference (not stored in config)
            config["_prompt_forced_llm_id"] = forced_llm_id

            return config


async def set_landing_registration_config(prompt_id: int, config: dict) -> bool:
    """
    Set the landing registration configuration for a prompt.
    Returns True on success, False on failure.
    """
    # Validate and sanitize config
    sanitized = {}

    # default_llm_id - must be integer or None
    if "default_llm_id" in config:
        val = config["default_llm_id"]
        sanitized["default_llm_id"] = int(val) if val not in (None, "", "null") else None

    # Boolean fields
    for field in ["public_prompts_access", "allow_file_upload", "allow_image_generation"]:
        if field in config:
            sanitized[field] = bool(config[field])

    # initial_balance - must be non-negative float
    if "initial_balance" in config:
        val = float(config.get("initial_balance", 0))
        sanitized["initial_balance"] = max(0.0, val)

    # billing_mode - must be one of the allowed values
    if "billing_mode" in config:
        val = config["billing_mode"]
        if val in ("user_pays", "manager_pays"):
            sanitized["billing_mode"] = val

    # billing_limit - must be positive float or None
    if "billing_limit" in config:
        val = config["billing_limit"]
        if val in (None, "", "null"):
            sanitized["billing_limit"] = None
        else:
            sanitized["billing_limit"] = max(0.0, float(val))

    # billing_limit_action - must be one of the allowed values
    if "billing_limit_action" in config:
        val = config["billing_limit_action"]
        if val in ("block", "notify", "auto_refill"):
            sanitized["billing_limit_action"] = val

    # billing_auto_refill_amount - must be positive float
    if "billing_auto_refill_amount" in config:
        val = config["billing_auto_refill_amount"]
        if val not in (None, "", "null"):
            sanitized["billing_auto_refill_amount"] = max(1.0, float(val))

    # billing_max_limit - must be positive float or None
    if "billing_max_limit" in config:
        val = config["billing_max_limit"]
        if val in (None, "", "null"):
            sanitized["billing_max_limit"] = None
        else:
            sanitized["billing_max_limit"] = max(0.0, float(val))

    # category_access - must be JSON array or None
    if "category_access" in config:
        val = config["category_access"]
        if val in (None, "", "null"):
            sanitized["category_access"] = None
        elif isinstance(val, list):
            # Ensure all elements are integers
            sanitized["category_access"] = [int(x) for x in val]
        elif isinstance(val, str):
            try:
                parsed = orjson.loads(val)
                if isinstance(parsed, list):
                    sanitized["category_access"] = [int(x) for x in parsed]
            except (orjson.JSONDecodeError, ValueError):
                pass

    try:
        config_json = orjson.dumps(sanitized).decode('utf-8')

        async with get_db_connection() as conn:
            await conn.execute(
                "UPDATE PROMPTS SET landing_registration_config = ? WHERE id = ?",
                (config_json, prompt_id)
            )
            await conn.commit()

        return True
    except Exception as e:
        logger.error(f"Error setting landing_registration_config for prompt {prompt_id}: {e}")
        return False


# =============================================================================
# Watchdog Configuration
# =============================================================================

VALID_WATCHDOG_MODES = ("interview", "coaching", "education", "custom")
VALID_THRESHOLD_KEYS = {
    "max_turns_off_topic", "max_turns_same_subtopic",
    "max_warnings_before_action", "max_unanswered_questions",
}
# Keys that always get a default value (original thresholds)
_REQUIRED_THRESHOLD_KEYS = {"max_turns_off_topic", "max_turns_same_subtopic"}

# --- Watchdog range constants (single source of truth) ---
# Used by validate_watchdog_config(), _sanitize_suggestion(), and _build_suggest_system_prompt()
WATCHDOG_FREQ_MIN, WATCHDOG_FREQ_MAX = 1, 20
WATCHDOG_HINT_MIN, WATCHDOG_HINT_MAX = 100, 2000
WATCHDOG_THRESHOLD_MIN, WATCHDOG_THRESHOLD_MAX = 0, 50
WATCHDOG_TAKEOVER_THRESHOLD_MIN, WATCHDOG_TAKEOVER_THRESHOLD_MAX = 1, 50
WATCHDOG_OBJ_MAX_CHARS = 500
WATCHDOG_OBJ_MAX_COUNT = 20
WATCHDOG_STEERING_MAX_CHARS = 5000


def _parse_watchdog_config_for_template(raw_json: Optional[str]) -> dict:
    """Parse watchdog_config from DB for template rendering.
    Returns nested defaults if NULL/invalid. Deep-merges per sub-config."""
    config = get_default_watchdog_config()
    if raw_json:
        try:
            stored = orjson.loads(raw_json)
            if isinstance(stored, dict):
                # Deep-merge: merge each sub-config independently
                if "pre_watchdog" in stored and isinstance(stored["pre_watchdog"], dict):
                    config["pre_watchdog"].update(stored["pre_watchdog"])
                if "post_watchdog" in stored and isinstance(stored["post_watchdog"], dict):
                    # Merge thresholds separately to avoid losing defaults
                    stored_post = stored["post_watchdog"]
                    if "thresholds" in stored_post and isinstance(stored_post["thresholds"], dict):
                        config["post_watchdog"]["thresholds"].update(stored_post["thresholds"])
                        stored_post_copy = {k: v for k, v in stored_post.items() if k != "thresholds"}
                        config["post_watchdog"].update(stored_post_copy)
                    else:
                        config["post_watchdog"].update(stored_post)
        except orjson.JSONDecodeError:
            pass
    return config


def get_default_watchdog_config() -> dict:
    """Factory function returning the nested pre/post watchdog defaults."""
    return {
        "pre_watchdog": {
            "enabled": False,
            "llm_id": None,
            "objectives": [],
            "steering_prompt": "",
            "frequency": 1,
            "can_takeover": True,
            "can_lock": False,
        },
        "post_watchdog": {
            "enabled": False,
            "llm_id": None,
            "mode": "custom",
            "objectives": [],
            "steering_prompt": "",
            "frequency": 3,
            "max_hint_chars": 500,
            "thresholds": {"max_turns_off_topic": 3, "max_turns_same_subtopic": 5},
            "can_takeover": False,
            "takeover_threshold": 5,
            "can_lock": False,
        },
    }


def _validate_pre_watchdog_config(config: dict) -> dict:
    """Validate the pre_watchdog sub-config. Returns sanitized dict or raises ValueError."""
    defaults = get_default_watchdog_config()["pre_watchdog"]
    sanitized = defaults.copy()

    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        enabled = bool(enabled)
    sanitized["enabled"] = enabled

    if not enabled:
        return sanitized

    # llm_id (required when enabled)
    llm_id = config.get("llm_id")
    if llm_id is None:
        raise ValueError("pre_watchdog.llm_id is required when enabled")
    try:
        sanitized["llm_id"] = int(llm_id)
    except (TypeError, ValueError):
        raise ValueError("pre_watchdog.llm_id must be an integer")

    # objectives (required when enabled)
    objectives = config.get("objectives", [])
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("pre_watchdog requires at least one objective when enabled")
    if len(objectives) > WATCHDOG_OBJ_MAX_COUNT:
        raise ValueError(f"pre_watchdog: maximum {WATCHDOG_OBJ_MAX_COUNT} objectives allowed")
    sanitized_objectives = []
    for obj in objectives:
        if not isinstance(obj, str):
            continue
        obj = obj.strip()[:WATCHDOG_OBJ_MAX_CHARS]
        if obj:
            sanitized_objectives.append(obj)
    if not sanitized_objectives:
        raise ValueError("pre_watchdog requires at least one non-empty objective")
    sanitized["objectives"] = sanitized_objectives

    # steering_prompt (optional)
    steering_prompt = config.get("steering_prompt", "")
    if isinstance(steering_prompt, str):
        sanitized["steering_prompt"] = steering_prompt.strip()[:WATCHDOG_STEERING_MAX_CHARS]

    # frequency (1-20, default 1)
    frequency = config.get("frequency", 1)
    try:
        frequency = int(frequency)
    except (TypeError, ValueError):
        frequency = 1
    if frequency < WATCHDOG_FREQ_MIN or frequency > WATCHDOG_FREQ_MAX:
        raise ValueError(f"pre_watchdog.frequency must be between {WATCHDOG_FREQ_MIN} and {WATCHDOG_FREQ_MAX}")
    sanitized["frequency"] = frequency

    # can_takeover (bool, default True)
    can_takeover = config.get("can_takeover", True)
    if not isinstance(can_takeover, bool):
        raise ValueError("pre_watchdog.can_takeover must be a boolean")
    sanitized["can_takeover"] = can_takeover

    # can_lock (bool, default False)
    can_lock = config.get("can_lock", False)
    if not isinstance(can_lock, bool):
        raise ValueError("pre_watchdog.can_lock must be a boolean")
    sanitized["can_lock"] = can_lock

    return sanitized


def _validate_post_watchdog_config(config: dict) -> dict:
    """Validate the post_watchdog sub-config. Returns sanitized dict or raises ValueError."""
    defaults = get_default_watchdog_config()["post_watchdog"]
    sanitized = defaults.copy()
    sanitized["thresholds"] = defaults["thresholds"].copy()

    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        enabled = bool(enabled)
    sanitized["enabled"] = enabled

    if not enabled:
        return sanitized

    # llm_id (required when enabled)
    llm_id = config.get("llm_id")
    if llm_id is None:
        raise ValueError("post_watchdog.llm_id is required when enabled")
    try:
        sanitized["llm_id"] = int(llm_id)
    except (TypeError, ValueError):
        raise ValueError("post_watchdog.llm_id must be an integer")

    # mode
    mode = config.get("mode", "custom")
    if mode not in VALID_WATCHDOG_MODES:
        raise ValueError(f"post_watchdog.mode must be one of: {', '.join(VALID_WATCHDOG_MODES)}")
    sanitized["mode"] = mode

    # objectives (required when enabled)
    objectives = config.get("objectives", [])
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("post_watchdog requires at least one objective when enabled")
    if len(objectives) > WATCHDOG_OBJ_MAX_COUNT:
        raise ValueError(f"post_watchdog: maximum {WATCHDOG_OBJ_MAX_COUNT} objectives allowed")
    sanitized_objectives = []
    for obj in objectives:
        if not isinstance(obj, str):
            continue
        obj = obj.strip()[:WATCHDOG_OBJ_MAX_CHARS]
        if obj:
            sanitized_objectives.append(obj)
    if not sanitized_objectives:
        raise ValueError("post_watchdog requires at least one non-empty objective")
    sanitized["objectives"] = sanitized_objectives

    # steering_prompt (optional)
    steering_prompt = config.get("steering_prompt", "")
    if isinstance(steering_prompt, str):
        sanitized["steering_prompt"] = steering_prompt.strip()[:WATCHDOG_STEERING_MAX_CHARS]

    # frequency
    frequency = config.get("frequency", 3)
    try:
        frequency = int(frequency)
    except (TypeError, ValueError):
        frequency = 3
    if frequency < WATCHDOG_FREQ_MIN or frequency > WATCHDOG_FREQ_MAX:
        raise ValueError(f"post_watchdog.frequency must be between {WATCHDOG_FREQ_MIN} and {WATCHDOG_FREQ_MAX}")
    sanitized["frequency"] = frequency

    # max_hint_chars
    max_hint_chars = config.get("max_hint_chars", 500)
    try:
        max_hint_chars = int(max_hint_chars)
    except (TypeError, ValueError):
        max_hint_chars = 500
    if max_hint_chars < WATCHDOG_HINT_MIN or max_hint_chars > WATCHDOG_HINT_MAX:
        raise ValueError(f"post_watchdog.max_hint_chars must be between {WATCHDOG_HINT_MIN} and {WATCHDOG_HINT_MAX}")
    sanitized["max_hint_chars"] = max_hint_chars

    # thresholds
    thresholds_raw = config.get("thresholds", {})
    if not isinstance(thresholds_raw, dict):
        thresholds_raw = {}
    sanitized_thresholds = {}
    for key in VALID_THRESHOLD_KEYS:
        val = thresholds_raw.get(key)
        if val is not None:
            try:
                val = int(val)
            except (TypeError, ValueError):
                default = sanitized["thresholds"].get(key)
                if default is not None:
                    val = default
                else:
                    continue
            if val < WATCHDOG_THRESHOLD_MIN or val > WATCHDOG_THRESHOLD_MAX:
                raise ValueError(f"post_watchdog threshold '{key}' must be between {WATCHDOG_THRESHOLD_MIN} and {WATCHDOG_THRESHOLD_MAX}")
            sanitized_thresholds[key] = val
        elif key in _REQUIRED_THRESHOLD_KEYS:
            sanitized_thresholds[key] = sanitized["thresholds"][key]
    sanitized["thresholds"] = sanitized_thresholds

    # can_takeover (bool, default False)
    can_takeover = config.get("can_takeover", False)
    if not isinstance(can_takeover, bool):
        raise ValueError("post_watchdog.can_takeover must be a boolean")
    sanitized["can_takeover"] = can_takeover

    # takeover_threshold (1-50, default 5)
    takeover_threshold = config.get("takeover_threshold", 5)
    try:
        takeover_threshold = int(takeover_threshold)
    except (TypeError, ValueError):
        takeover_threshold = 5
    if takeover_threshold < WATCHDOG_TAKEOVER_THRESHOLD_MIN or takeover_threshold > WATCHDOG_TAKEOVER_THRESHOLD_MAX:
        raise ValueError(f"post_watchdog.takeover_threshold must be between {WATCHDOG_TAKEOVER_THRESHOLD_MIN} and {WATCHDOG_TAKEOVER_THRESHOLD_MAX}")
    sanitized["takeover_threshold"] = takeover_threshold

    # can_lock (bool, default False)
    can_lock = config.get("can_lock", False)
    if not isinstance(can_lock, bool):
        raise ValueError("post_watchdog.can_lock must be a boolean")
    sanitized["can_lock"] = can_lock

    return sanitized


def validate_watchdog_config(config: dict) -> dict:
    """Validate and sanitize nested watchdog config. Sync, structural only (no DB checks).
    Accepts nested {pre_watchdog: {}, post_watchdog: {}} format.
    Returns sanitized nested dict or raises ValueError."""
    if not isinstance(config, dict):
        raise ValueError("watchdog_config must be a JSON object")

    pre_raw = config.get("pre_watchdog", {})
    post_raw = config.get("post_watchdog", {})

    if not isinstance(pre_raw, dict):
        pre_raw = {}
    if not isinstance(post_raw, dict):
        post_raw = {}

    return {
        "pre_watchdog": _validate_pre_watchdog_config(pre_raw),
        "post_watchdog": _validate_post_watchdog_config(post_raw),
    }


async def get_watchdog_config(prompt_id: int) -> dict:
    """Read nested watchdog config from DB, deep-merged with defaults."""
    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT watchdog_config FROM PROMPTS WHERE id = ?",
                (prompt_id,)
            )
            result = await cursor.fetchone()

            if not result:
                raise HTTPException(status_code=404, detail="Prompt not found")

            config = get_default_watchdog_config()
            config_json = result[0]
            if config_json:
                try:
                    stored = orjson.loads(config_json)
                    if isinstance(stored, dict):
                        if "pre_watchdog" in stored and isinstance(stored["pre_watchdog"], dict):
                            config["pre_watchdog"].update(stored["pre_watchdog"])
                        if "post_watchdog" in stored and isinstance(stored["post_watchdog"], dict):
                            stored_post = stored["post_watchdog"]
                            if "thresholds" in stored_post and isinstance(stored_post["thresholds"], dict):
                                config["post_watchdog"]["thresholds"].update(stored_post["thresholds"])
                                rest = {k: v for k, v in stored_post.items() if k != "thresholds"}
                                config["post_watchdog"].update(rest)
                            else:
                                config["post_watchdog"].update(stored_post)
                except orjson.JSONDecodeError:
                    logger.warning("Invalid JSON in watchdog_config for prompt %d", prompt_id)

            return config


async def set_watchdog_config(prompt_id: int, config: dict) -> bool:
    """Validate, check FK llm_ids, serialize and store nested watchdog config.
    Returns True on success. Raises ValueError on validation error."""
    sanitized = validate_watchdog_config(config)

    # Async FK validation: check llm_id exists in LLM table for both sub-configs
    for sub_key in ("pre_watchdog", "post_watchdog"):
        sub_cfg = sanitized.get(sub_key, {})
        if sub_cfg.get("enabled") and sub_cfg.get("llm_id") is not None:
            async with get_db_connection(readonly=True) as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT id FROM LLM WHERE id = ?", (sub_cfg["llm_id"],))
                    if not await cursor.fetchone():
                        raise ValueError(f"LLM with id {sub_cfg['llm_id']} does not exist ({sub_key})")

    config_json = orjson.dumps(sanitized).decode("utf-8")

    async with get_db_connection() as conn:
        await conn.execute(
            "UPDATE PROMPTS SET watchdog_config = ? WHERE id = ?",
            (config_json, prompt_id)
        )
        await conn.commit()

    return True


# =============================================================================
# Watchdog AI Auto-fill (suggest config via LLM)
# =============================================================================

def _build_suggest_system_prompt() -> str:
    """Build the system prompt for watchdog config suggestion.
    Ranges shown to the LLM are derived from the real constants."""
    modes_str = ", ".join(f'"{m}"' for m in VALID_WATCHDOG_MODES)
    return f"""You are an expert AI configuration assistant. Your task is to analyze an AI prompt/personality and suggest optimal Watchdog configuration values.

The Watchdog is a silent conversation supervisor that runs in the background. It never speaks to the user. It evaluates the conversation against objectives and provides internal steering hints to the main AI.

Based on the prompt provided, return a JSON object with these fields:

1. "mode": One of {modes_str}.
   - "interview": The prompt conducts interviews, collects information, follows a structured plan.
   - "coaching": The prompt guides personal/professional development, tracks goals.
   - "education": The prompt teaches, tutors, tracks learning objectives.
   - "custom": General purpose or doesn't fit other categories.

2. "objectives": Array of 3-7 specific monitoring objectives (strings, each max {WATCHDOG_OBJ_MAX_CHARS} chars).
   Write them as clear instructions for a supervisor. Examples:
   - "Track which topics from the interview plan have been covered"
   - "Detect when the user changes subject away from the learning goal"
   - "Flag if the AI gives contradictory advice across turns"

3. "steering_prompt": Custom instructions for the watchdog evaluator (max {WATCHDOG_STEERING_MAX_CHARS} chars).
   Write as if instructing a silent supervisor who analyzes conversation transcripts.
   Be specific to the prompt's domain and purpose.

4. "frequency": Integer {WATCHDOG_FREQ_MIN}-{WATCHDOG_FREQ_MAX}. How often to evaluate (every N user turns).
   - 1-2: High-stakes conversations needing constant monitoring.
   - 3-4: Standard conversations with moderate oversight.
   - 5+: Relaxed conversations where occasional checks suffice.

5. "thresholds": Object with (include only keys relevant to the prompt):
   - "max_turns_off_topic": Integer {WATCHDOG_THRESHOLD_MIN}-{WATCHDOG_THRESHOLD_MAX}. Turns before flagging drift. Typical: 2-5. ALWAYS include.
   - "max_turns_same_subtopic": Integer {WATCHDOG_THRESHOLD_MIN}-{WATCHDOG_THRESHOLD_MAX}. Turns before flagging repetition. Typical: 3-8. ALWAYS include.
   - "max_warnings_before_action": Integer {WATCHDOG_THRESHOLD_MIN}-{WATCHDOG_THRESHOLD_MAX}. How many warnings/ultimatums before the AI must act decisively. Typical: 1-2 for strict/authority roles, 3-5 for lenient. Include for roles that issue warnings.
   - "max_unanswered_questions": Integer {WATCHDOG_THRESHOLD_MIN}-{WATCHDOG_THRESHOLD_MAX}. Consecutive essential questions the user can dodge before flagging. Typical: 1-3. Include for interview/structured roles.

6. "max_hint_chars": Integer {WATCHDOG_HINT_MIN}-{WATCHDOG_HINT_MAX}. Hint length based on complexity.
   - Simple prompts: 200-300
   - Complex multi-phase prompts: 500-1000

IMPORTANT: The watchdog system includes a built-in Role Coherence module that automatically monitors: commitment follow-through (verbal conclusions without blocking, unfulfilled ultimatums), strictness calibration (too lenient/harsh for the role), tool usage logic, and user good-faith assessment. Your objectives and steering_prompt should focus on DOMAIN-SPECIFIC monitoring for this particular prompt. Do NOT include generic objectives about "maintaining role" or "using tools correctly" - those are covered automatically.

Respond ONLY with a valid JSON object. No markdown fences, no extra text."""


def _sanitize_suggestion(suggestion: dict) -> dict:
    """Sanitize AI suggestion to ensure safe ranges. Best-effort clamp, not strict validation.
    Uses the same range constants as validate_watchdog_config() to avoid divergence."""
    defaults = get_default_watchdog_config()

    mode = suggestion.get("mode", defaults["mode"])
    if mode not in VALID_WATCHDOG_MODES:
        mode = defaults["mode"]

    objectives = suggestion.get("objectives", [])
    if not isinstance(objectives, list):
        objectives = []
    objectives = [
        str(o).strip()[:WATCHDOG_OBJ_MAX_CHARS]
        for o in objectives if o and str(o).strip()
    ][:WATCHDOG_OBJ_MAX_COUNT]

    steering_prompt = str(suggestion.get("steering_prompt", "")).strip()[:WATCHDOG_STEERING_MAX_CHARS]

    frequency = suggestion.get("frequency", defaults["frequency"])
    try:
        frequency = max(WATCHDOG_FREQ_MIN, min(WATCHDOG_FREQ_MAX, int(frequency)))
    except (TypeError, ValueError):
        frequency = defaults["frequency"]

    max_hint_chars = suggestion.get("max_hint_chars", defaults["max_hint_chars"])
    try:
        max_hint_chars = max(WATCHDOG_HINT_MIN, min(WATCHDOG_HINT_MAX, int(max_hint_chars)))
    except (TypeError, ValueError):
        max_hint_chars = defaults["max_hint_chars"]

    thresholds = suggestion.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}

    def _clamp_threshold(key: str):
        val = thresholds.get(key)
        if val is None:
            default = defaults["thresholds"].get(key)
            return default  # None for optional keys not provided
        try:
            return max(WATCHDOG_THRESHOLD_MIN, min(WATCHDOG_THRESHOLD_MAX, int(val)))
        except (TypeError, ValueError):
            return defaults["thresholds"].get(key)

    clamped_thresholds = {}
    for key in VALID_THRESHOLD_KEYS:
        val = _clamp_threshold(key)
        if val is not None:
            clamped_thresholds[key] = val

    return {
        "mode": mode,
        "objectives": objectives,
        "steering_prompt": steering_prompt,
        "frequency": frequency,
        "max_hint_chars": max_hint_chars,
        "thresholds": clamped_thresholds,
    }


@router.post("/api/watchdog/suggest-config")
async def watchdog_suggest_config(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if not await current_user.is_admin and not await current_user.is_manager:
        raise HTTPException(status_code=403, detail="Access denied")

    body = await request.json()
    llm_id = body.get("llm_id")
    prompt_text = body.get("prompt_text", "").strip()

    # Validations
    if not llm_id:
        raise HTTPException(status_code=400, detail="llm_id is required")
    if not prompt_text or len(prompt_text) < 10:
        raise HTTPException(status_code=400, detail="prompt_text is required (min 10 chars)")

    prompt_text = prompt_text[:10000]  # Truncate to limit cost

    # Lookup LLM
    async with get_db_connection(readonly=True) as conn:
        cursor = await conn.execute(
            "SELECT machine, model FROM LLM WHERE id = ?", (llm_id,)
        )
        llm_row = await cursor.fetchone()
        if not llm_row:
            raise HTTPException(status_code=404, detail="LLM not found")

    machine, model = llm_row[0], llm_row[1]

    # Resolve BYOK/system key for this provider
    user_api_keys = {}
    try:
        async with get_db_connection(readonly=True) as conn:
            cursor = await conn.execute(
                "SELECT user_api_keys FROM USER_DETAILS WHERE user_id = ?",
                (current_user.id,),
            )
            row = await cursor.fetchone()
            if row and row[0]:
                decrypted = decrypt_api_key(row[0])
                if decrypted:
                    parsed = orjson.loads(decrypted)
                    if isinstance(parsed, dict):
                        user_api_keys = parsed
    except Exception:
        logger.warning(
            "Watchdog suggest-config: failed reading user API keys for user=%d",
            current_user.id,
            exc_info=True,
        )

    api_key_mode = await get_user_api_key_mode(current_user.id)
    resolved_key, use_system = resolve_api_key_for_provider(
        user_api_keys,
        api_key_mode,
        machine,
    )
    if not resolved_key and not use_system:
        raise HTTPException(
            status_code=400,
            detail=f"API key required for provider {machine} in own-only mode.",
        )

    # Call LLM
    from tools.llm_caller import (
        call_llm_non_streaming_with_usage,
        extract_json_from_llm_response,
    )

    system_prompt = _build_suggest_system_prompt()
    user_message = f"Analyze this AI prompt and suggest watchdog configuration:\n\n{prompt_text}"

    try:
        result = await call_llm_non_streaming_with_usage(
            machine=machine,
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            timeout=45,
            max_tokens=1500,
            api_key_override=resolved_key,
        )
        raw_response = result.text
    except Exception:
        logger.exception("Watchdog suggest-config: LLM call failed (llm_id=%s)", llm_id)
        raise HTTPException(status_code=502, detail="AI analysis failed. Please try again or configure manually.")

    # Bill this non-streaming LLM call with returned usage
    try:
        async with get_db_connection() as bill_conn:
            await bill_conn.execute("BEGIN IMMEDIATE")
            bill_cursor = await bill_conn.cursor()
            input_cost, output_cost = await get_llm_token_costs(model, bill_conn)
            billed = await consume_token(
                user_id=current_user.id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                input_token_cost_per_million=input_cost,
                output_token_cost_per_million=output_cost,
                conn=bill_conn,
                cursor=bill_cursor,
                prompt_id=None,
            )
            if not billed:
                logger.warning(
                    "Watchdog suggest-config: consume_token returned False (user=%d, model=%s, in=%d, out=%d)",
                    current_user.id,
                    model,
                    result.input_tokens,
                    result.output_tokens,
                )
            await bill_conn.commit()
    except Exception:
        logger.exception(
            "Watchdog suggest-config: billing failed (user=%d, llm_id=%s)",
            current_user.id,
            llm_id,
        )

    # Parse JSON from response
    suggestion = extract_json_from_llm_response(raw_response)
    if not suggestion:
        raise HTTPException(
            status_code=502,
            detail="The AI returned an invalid response. Try again or configure manually."
        )

    # Sanitize and validate ranges
    sanitized = _sanitize_suggestion(suggestion)

    if not sanitized["objectives"]:
        raise HTTPException(
            status_code=502,
            detail="The AI could not generate valid objectives. Please configure manually."
        )

    return sanitized


async def get_prompt_owner_id(prompt_id: int) -> Optional[int]:
    """
    Get the owner user ID for a prompt.
    Returns the user_id of the owner or None if not found.
    """
    async with get_db_connection(readonly=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT user_id FROM PROMPT_PERMISSIONS WHERE prompt_id = ? AND permission_level = 'owner'",
                (prompt_id,)
            )
            result = await cursor.fetchone()
            return result[0] if result else None
