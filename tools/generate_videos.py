# tools/generate_videos.py

import os
import orjson
import asyncio
import aiohttp
import dramatiq
import time
import re
import base64
from dotenv import load_dotenv
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Response, HTTPException, Depends, Request, Form, status, UploadFile, File
from urllib.parse import urlparse

# Own libraries
from rediscfg import redis_client, broker
from common import estimate_message_tokens, Cost
from models import User, ConnectionManager
from whatsapp import is_whatsapp_conversation
from tools import register_tool, register_dramatiq_task, register_function_handler
from save_images import save_image_locally, generate_img_token, resize_image
from auth import hash_password, verify_password, get_user_by_username, get_current_user, create_access_token, get_user_by_id, get_user_from_phone_number
from auth import get_current_user_from_websocket, get_user_id_from_conversation, get_user_by_token, create_user_info, create_login_response, generate_magic_link
from common import Cost, generate_user_hash, has_sufficient_balance, cost_tts, cache_directory, users_directory, elevenlabs_key, openai_key, tts_engine, get_balance, deduct_balance, record_daily_usage, load_service_costs, ALGORITHM, estimate_message_tokens, CLOUDFLARE_BASE_URL, MEDIA_TOKEN_EXPIRE_HOURS

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_KEY')
VIDEO_GENERATION_ENGINE = os.getenv('VIDEO_GENERATION_ENGINE', 'veo-3').lower()
VIDEO_GENERATION_TIMEOUT = int(os.getenv('VIDEO_GENERATION_TIMEOUT', 600))  # Timeout in seconds, default 10 minutes

# VEO Model Configuration
# Available models (Gemini API):
#   - veo-3.1-fast-generate-preview (fast, good for most use cases)
#   - veo-3.1-generate-preview (best quality, slower)
#   - veo-2.0-generate-001 (stable, older)
VEO_MODEL = os.getenv('VEO_MODEL', 'veo-3.1-fast-generate-preview')

async def generate_video_veo3(prompt: str, aspect_ratio: str = "16:9", negative_prompt: str = None) -> tuple:
    """Generate video using Google's VEO-3 model with official google-genai library"""
    try:
        # Import google-genai library
        from google import genai
        from google.genai import types
        
        # Create client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Create generation prompt
        generation_prompt = f"Create a high-quality 8-second video: {prompt}"
        
        # Create config with available parameters
        config_params = {}
        if negative_prompt:
            config_params["negative_prompt"] = negative_prompt
            
        config = types.GenerateVideosConfig(**config_params)
        
        print(f"Starting VEO-3 video generation with prompt: {generation_prompt}")
        
        # Start video generation operation
        operation = client.models.generate_videos(
            model=VEO_MODEL,
            prompt=generation_prompt,
            config=config
        )
        
        print(f"Video generation operation started: {operation.name}")
        
        # Poll for completion with progress updates
        max_wait_time = 360  # 6 minutes in seconds
        poll_interval = 10   # 10 seconds between polls
        elapsed_time = 0
        
        while not operation.done:
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval
            
            # Update operation status
            operation = client.operations.get(operation)
            
            print(f"Video generation in progress... {elapsed_time}s elapsed (max {max_wait_time}s)")
            
            if elapsed_time >= max_wait_time:
                raise Exception("Video generation timed out after 6 minutes")
        
        # Get the generated video
        if hasattr(operation, 'result') and hasattr(operation.result, 'generated_videos'):
            generated_video = operation.result.generated_videos[0]
            
            # Download video data
            video_file = generated_video.video
            video_data = client.files.download(file=video_file)
            
            # Determine mime type
            mime_type = "video/mp4"  # VEO-3 typically generates MP4
            
            print(f"Video generation completed! Size: {len(video_data)} bytes")
            
            return video_data, prompt, aspect_ratio, mime_type
        else:
            raise Exception("No video generated in operation result")
            
    except ImportError:
        raise Exception("google-genai library not installed. Install with: pip install google-genai")
    except Exception as e:
        if "google-genai" in str(e):
            raise Exception(f"Google GenAI library error: {str(e)}")
        else:
            raise Exception(f"VEO-3 generation failed: {str(e)}")

async def generate_video(prompt: str) -> tuple:
    print(f"Generating video using {VIDEO_GENERATION_ENGINE}")
    
    # Extract aspect ratio from prompt if specified
    aspect_ratio_match = re.search(r'\b(\d+:\d+)\b', prompt)
    if aspect_ratio_match:
        aspect_ratio = aspect_ratio_match.group(1)
        prompt = re.sub(r'\b\d+:\d+\b', '', prompt).strip()
    else:
        aspect_ratio = "16:9"

    # Extract negative prompt if specified
    negative_prompt = None
    negative_match = re.search(r'(?:without|avoid|exclude|no)\s+([^.!?]+)', prompt, re.IGNORECASE)
    if negative_match:
        negative_prompt = negative_match.group(1).strip()

    if VIDEO_GENERATION_ENGINE == 'veo-3':
        return await generate_video_veo3(prompt, aspect_ratio, negative_prompt)
    else:
        raise ValueError(f"Unsupported video generation engine: {VIDEO_GENERATION_ENGINE}")

async def save_video_locally(video_data, filename, user, conversation_id, source="bot", format="mp4"):
    """Save video file locally using the same structure as images"""
    try:
        # Use the same directory structure as images
        from common import users_directory
        
        # Generate the user hash
        hash_prefix1, hash_prefix2, user_hash = generate_user_hash(user.username)

        # Create the conversation prefixes according to the specified format
        conversation_id_str = f"{conversation_id:07d}"
        conversation_id_prefix1 = conversation_id_str[:3]
        conversation_id_prefix2 = conversation_id_str[3:]
        
        # Build the directory path (video instead of img)
        file_location = os.path.join(users_directory, hash_prefix1, hash_prefix2, user_hash, "files", conversation_id_prefix1, conversation_id_prefix2, "video", source)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(file_location):
            os.makedirs(file_location)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        base_filename = f"{filename}_{timestamp}"
        
        # Save the video file
        file_path = os.path.join(file_location, f"{base_filename}.{format}")
        with open(file_path, 'wb') as f:
            f.write(video_data)
        
        # Generate video path (same structure as images)
        video_path = f"users/{hash_prefix1}/{hash_prefix2}/{user_hash}/files/{conversation_id_prefix1}/{conversation_id_prefix2}/video/{source}/{base_filename}.{format}"
        
        # Generate URLs (same as images - no token in database)
        base_url = f"{CLOUDFLARE_BASE_URL}{video_path}"
        token_url = base_url  # Same as base_url, process_message will add token when serving
        
        print(f"Video saved: {file_path}")
        print(f"Video URL: {token_url}")
        
        return base_url, token_url
        
    except Exception as e:
        print(f"Error saving video: {e}")
        raise e

async def generate_video_task(channel_name: str, prompt: str, conversation_id: int, user_id: int, is_whatsapp: bool, request_url: str):
    try:
        print("Entering generate_video_task")
        
        # Send initial status update
        await redis_client.publish(channel_name, orjson.dumps({
            'progress_update': 'Starting video generation...'
        }).decode())
        
        # Custom generate_video_veo3_with_progress function
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=GEMINI_API_KEY)
            generation_prompt = f"Create a high-quality 8-second video: {prompt}"
            
            config = types.GenerateVideosConfig()
            
            print(f"Starting VEO-3 video generation with prompt: {generation_prompt}")
            
            # Start video generation operation
            operation = client.models.generate_videos(
                model=VEO_MODEL,
                prompt=generation_prompt,
                config=config
            )
            
            await redis_client.publish(channel_name, orjson.dumps({
                'progress_update': 'Video generation started'
            }).decode())
            
            # Poll for completion with progress updates
            max_wait_time = 360  # 6 minutes in seconds
            poll_interval = 10   # 10 seconds between polls
            elapsed_time = 0
            
            while not operation.done:
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
                
                # Update operation status
                operation = client.operations.get(operation)
                
                # Send progress update to chat
                progress_msg = f"Generating video... {elapsed_time}s elapsed (up to 6min)"
                await redis_client.publish(channel_name, orjson.dumps({
                    'progress_update': progress_msg
                }).decode())
                
                print(f"Video generation in progress... {elapsed_time}s elapsed")
                
                if elapsed_time >= max_wait_time:
                    raise Exception("Video generation timed out after 6 minutes")
            
            # Get the generated video
            if hasattr(operation, 'result') and hasattr(operation.result, 'generated_videos'):
                generated_video = operation.result.generated_videos[0]
                video_file = generated_video.video
                video_data = client.files.download(file=video_file)
                mime_type = "video/mp4"
                
                print(f"Video generation completed! Size: {len(video_data)} bytes")
            else:
                raise Exception("No video generated in operation result")
            
        except Exception as e:
            raise Exception(f"VEO-3 generation failed: {str(e)}")
        
        filename = f"generated_video_{conversation_id}_{int(time.time())}"
        source = "bot"
        format = "mp4"

        user = await get_user_by_id(user_id)

        await redis_client.publish(channel_name, orjson.dumps({
            'progress_update': 'Saving video...'
        }).decode())

        video_link_base, video_link_token = await save_video_locally(
            video_data=video_data,
            filename=filename,
            user=user,
            conversation_id=conversation_id,
            source=source,
            format=format
        )

        # Generate token for immediate display (same as process_message does for DB loads)
        from datetime import datetime, timedelta
        expiration = datetime.utcnow() + timedelta(hours=MEDIA_TOKEN_EXPIRE_HOURS)
        token = generate_img_token(video_link_base, expiration, user)
        video_url_with_token = f"{video_link_base}?token={token}"

        # Use the same JSON structure for both display and save
        video_content = orjson.dumps([
            {
                "type": "video_url",
                "video_url": {
                    "url": video_url_with_token,  # Use URL with token for display
                    "alt": prompt,
                    "mime_type": mime_type
                }
            }
        ]).decode()

        content_to_show = video_content
        content_to_save = orjson.dumps([
            {
                "type": "video_url",
                "video_url": {
                    "url": video_link_base,  # Use base URL for storage
                    "alt": prompt,
                    "mime_type": mime_type
                }
            }
        ]).decode()

        await redis_client.publish(channel_name, orjson.dumps({
            'content_to_show': content_to_show,
            'content_to_save': content_to_save
        }).decode())
        
        await redis_client.publish(channel_name, 'END')
        
    except Exception as e:
        print(f"Error in generate_video_task: {e}")
        await redis_client.publish(channel_name, orjson.dumps({'error': str(e)}).decode())
        await redis_client.publish(channel_name, 'END')

@dramatiq.actor
def generate_video_task_actor(channel_name: str, prompt: str, conversation_id: int, user_id: int, is_whatsapp: bool, request_url: str):
    asyncio.run(generate_video_task(channel_name, prompt, conversation_id, user_id, is_whatsapp, request_url))

async def handle_generate_video(function_arguments, messages, model, temperature, max_tokens, content, conversation_id, current_user, request, input_tokens, output_tokens, total_tokens, message_id, user_id, client, prompt, user_message=None):
    try:
        print("Entering handle_generate_video")
        video_prompt = function_arguments['prompt']
        channel_name = f"generate_video_response_{conversation_id}_{user_id}_{int(time.time())}"
        
        is_whatsapp = await is_whatsapp_conversation(conversation_id)
        request_url = str(request.url)
        
        # Video generation costs more than images
        video_cost = Cost.IMAGE_GENERATION_COST * 10  # Adjust cost as needed
        
        if not await has_sufficient_balance(user_id, video_cost):
            yield f"data: {orjson.dumps({'content': 'Insufficient balance to generate video.', 'save_to_db': True, 'yield': True}).decode()}\n\n"
            return

        generate_video_task_actor.send(channel_name, video_prompt, conversation_id, user_id, is_whatsapp, request_url)
        
        yield f"data: {orjson.dumps({'content': 'Generating video... This may take up to 6 minutes.', 'save_to_db': False, 'yield': True}).decode()}\n\n"
        
        async with redis_client.pubsub() as pubsub:
            await pubsub.subscribe(channel_name)
            start_time = time.time()
            while True:
                if time.time() - start_time > VIDEO_GENERATION_TIMEOUT:
                    yield f"data: {orjson.dumps({'content': 'Video generation timed out. Please try again.', 'save_to_db': True, 'yield': True}).decode()}\n\n"
                    break
                
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    data = message['data']
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    if data == 'END':
                        break
                    json_data = orjson.loads(data)
                    if 'error' in json_data:
                        yield f"data: {orjson.dumps({'content': json_data['error'], 'save_to_db': True, 'yield': True}).decode()}\n\n"
                    elif 'progress_update' in json_data:
                        # Send progress update without saving to DB
                        yield f"data: {orjson.dumps({'content': json_data['progress_update'], 'save_to_db': False, 'yield': True, 'replace_last': True}).decode()}\n\n"
                    elif 'content_to_show' in json_data and 'content_to_save' in json_data:
                        # Send video as separate type for proper rendering
                        yield f"data: {orjson.dumps({'video_content': json_data['content_to_show'], 'save_to_db': False, 'yield': True}).decode()}\n\n"
                        yield f"data: {orjson.dumps({'content': json_data['content_to_save'], 'save_to_db': True, 'yield': False}).decode()}\n\n"
                else:
                    await asyncio.sleep(0.1)
        
        await deduct_balance(user_id, video_cost)
        await record_daily_usage(
            user_id=user_id,
            usage_type='video',
            cost=video_cost,
            units=1
        )

    except Exception as e:
        print(f"Error in handle_generate_video: {e}")
        yield f"data: {orjson.dumps({'content': f'Error generating video: {str(e)}', 'save_to_db': True, 'yield': True}).decode()}\n\n"

# Register the tool for semantic router
register_tool({
    "type": "function",
    "function": {
        "name": "generateVideo",
        "description": f"Generate a high-quality 8-second video with audio using {VIDEO_GENERATION_ENGINE.upper()} (Google VEO-3) based on the provided prompt. You can specify aspect ratio (e.g., 16:9, 9:16, 1:1) and negative prompts in the description.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to generate the video, optionally including desired aspect ratio and elements to avoid"
                }
            },
            "required": ["prompt"],
            "additionalProperties": False
        }
    },
    "strict": True
})

# Register the Dramatiq task
register_dramatiq_task("generate_video_task_actor", generate_video_task_actor)

# Register the function handler
register_function_handler("generateVideo", handle_generate_video)