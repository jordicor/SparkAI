import html
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import orjson
import os
import hashlib
import io
from pydub import AudioSegment
import asyncio
import aiohttp
from fastapi import WebSocket
import string
import aiofiles
from database import get_db_connection

# Own libraries
from models import User, ConnectionManager
from tools.tts_load_balancer import get_elevenlabs_key
from common import Cost, has_sufficient_balance, cost_tts, cache_directory, elevenlabs_key, openai_key, tts_engine
from log_config import logger

manager = ConnectionManager()

def process_plain_text(text):
    #logger.debug(f"entra en process_plain_text: {text}")
    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML
    soup = BeautifulSoup(text, 'html.parser')

    # Replace images
    for img in soup.find_all('img'):
        img.replace_with("[IMAGE]")

    # Replace code
    for code in soup.find_all('code'):
        code.replace_with("[CODE]")

    # Replace links
    for a in soup.find_all('a'):
        href = a.get('href', '')
        domain = urlparse(href).netloc
        a.replace_with(f"[LINK, {domain}]" if domain else "[LINK]")
    
    # Get the clean text
    clean_text = soup.get_text()

    # Process Markdown
    # Replace Markdown links including the domain
    def replace_markdown_link(match):
        url = match.group(2)
        domain = urlparse(url).netloc
        return f"[LINK, {domain}]" if domain else "[LINK]"
    
    clean_text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', replace_markdown_link, clean_text)

    # Replace Markdown images
    clean_text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '[IMAGE]', clean_text)

    # Replace plain URLs in the text
    def replace_plain_url(match):
        url = match.group(0)
        domain = urlparse(url).netloc
        return f"[LINK, {domain}]" if domain else "[LINK]"
    
    clean_text = re.sub(r'https?://\S+', replace_plain_url, clean_text)
    
    return clean_text


async def insert_tts_break(text, min_length=50, max_length=90, look_ahead=30):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Define allowed characters (letters, digits, whitespace, punctuation, and accented characters)
    #allowed_characters = string.ascii_letters + string.digits + string.whitespace + string.punctuation + "áéíóúÁÉÍÓÚüÜñÑ"
    
    # Remove emojis and other unwanted characters
    #pattern = f"[^{re.escape(allowed_characters)}]"
    #text = re.sub(pattern, "", text)

    # Check if text length is less than or equal to max_length
    if len(text) <= max_length:
        return [text]

    chunks = []
    
    while len(text) > max_length:
        # Find last space or punctuation mark between min_length and max_length
        match = re.search(r".*[,;!?.)\]]\s|.*\s", text[min_length:max_length])

        if match:
            split_pos = match.end() + min_length

            # Look ahead to find a better split point
            look_ahead_match = re.search(r"[,;!?.)\]]", text[split_pos:split_pos+look_ahead])
            if look_ahead_match:
                split_pos = look_ahead_match.end() + split_pos

            chunk = text[:split_pos].rstrip()
            if len(chunk) >= min_length:
                chunks.append(chunk)
                text = text[split_pos:].lstrip()
            else:
                # If chunk is too short, find next split point
                next_match = re.search(r".*[,;!?.)\]]\s|.*\s", text[split_pos:])
                if next_match:
                    next_split_pos = next_match.end() + split_pos
                    chunks.append(text[:next_split_pos].rstrip())
                    text = text[next_split_pos:].lstrip()
                else:
                    # If no more split points, add all remaining text
                    chunks.append(text)
                    text = ""
        else:
            # If no suitable split point found, cut at max_length
            chunks.append(text[:max_length])
            text = text[max_length:]

    if text:
        # Handle remaining text
        if len(text) < min_length and chunks:
            chunks[-1] += " " + text
        else:
            chunks.append(text)

    return chunks

def process_text_for_tts(message):
    try:
        # Try to parse message as JSON
        message_data = orjson.loads(message)
        
        # Check if message_data is a list/dict that we can iterate over
        if isinstance(message_data, (list, dict)):
            processed_parts = []
            # If it's a dict, wrap it in a list to process it
            items = message_data if isinstance(message_data, list) else [message_data]
            
            for item in items:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        processed_parts.append(process_plain_text(item['text']))
                    elif item.get('type') == 'image_url':
                        processed_parts.append("IMAGE")
                    elif item.get('type') == 'audio_url':
                        processed_parts.append("AUDIO")
                    # Add more types as needed
                    # elif item['type'] == 'pdf_url':
                    #     processed_parts.append("PDF")
                    # elif item['type'] == 'doc_url':
                    #     processed_parts.append("DOCUMENT")
            
            return ' '.join(processed_parts)
        else:
            # If message_data is not iterable, treat it as plain text
            return process_plain_text(str(message_data))

    except orjson.JSONDecodeError:
        # If not JSON, treat as plain text
        return process_plain_text(message)
    except Exception as e:
        # Log the error and return the message as plain text
        logger.error(f"Error processing text for TTS: {str(e)}")
        return process_plain_text(str(message))


async def get_voice_code_from_prompt(prompt_id: int) -> str:
    async with get_db_connection(readonly=True) as conn:
        async with conn.execute('''
            SELECT v.voice_code
            FROM Prompts p
            JOIN voices v ON p.voice_id = v.id
            WHERE p.id = ?
        ''', (prompt_id,)) as cursor:
            result = await cursor.fetchone()
        if result:
            return result[0]
        else:
            raise ValueError("Prompt ID not found or voice code not available")

async def get_voice_code_from_conversation(conversation_id: int, current_user: User) -> str:
    async with get_db_connection(readonly=True) as conn:
        # First, check if the user has access to the conversation
        async with conn.execute('''
            SELECT c.id
            FROM CONVERSATIONS c
            WHERE c.id = ? AND c.user_id = ?
        ''', (conversation_id, current_user.id)) as cursor:
            conversation = await cursor.fetchone()
        
        if not conversation:
            raise ValueError("User does not have access to this conversation or conversation does not exist")

        # If the user has access, fetch the voice code
        async with conn.execute('''
            SELECT v.voice_code
            FROM CONVERSATIONS c
            JOIN PROMPTS p ON c.role_id = p.id
            JOIN VOICES v ON p.voice_id = v.id
            WHERE c.id = ?
        ''', (conversation_id,)) as cursor:
            result = await cursor.fetchone()
        
        if result:
            return result[0]
        else:
            raise ValueError("Voice code not available for this conversation")


def get_tts_generator(engine: str, voice_id: str, chunks: list):
    logger.info("entra en get_tts_generator")
    if engine == 'elevenlabs':
        logger.info("devuelve elevenlabs")
        return elevenlabs_generator(voice_id, chunks)
    elif engine == 'openai':
        return openai_generator(voice_id, chunks)
    elif engine == 'deepgram':
        return deepgram_generator(voice_id, chunks)
    else:
        raise ValueError(f"Unsupported TTS engine: {engine}")


async def send_cached_audio(websocket: WebSocket, full_path_opus: str):
    logger.debug(f"Sending cached audio: {full_path_opus}")
    with open(full_path_opus, "rb") as f:
        while chunk := f.read(16384):
            await manager.send_bytes(websocket, chunk)


def get_file_path(hash_digest):
    file_path = os.path.join(str(cache_directory), hash_digest[:2], hash_digest[2:4], hash_digest[4:6], hash_digest[6:8])
    file_name_opus = f"{hash_digest}.opus"
    full_path_opus = os.path.join(file_path, file_name_opus)
    return file_path, full_path_opus

async def elevenlabs_generator(voice_id: str, chunks: list):
    logger.info("Entering elevenlabs generator")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    async with aiohttp.ClientSession() as session:
        for i, chunk_text in enumerate(chunks):
            try:
                # Get a new API key for each chunk
                elevenlabs_key = get_elevenlabs_key()
                if not elevenlabs_key:
                    raise ValueError("No valid Elevenlabs API key available")

                headers = {
                    "Content-Type": "application/json",
                    "xi-api-key": elevenlabs_key
                }

                previous_text = None if i == 0 else " ".join(chunks[:i])
                next_text = None if i == len(chunks) - 1 else " ".join(chunks[i + 1:])
                payload = {
                    "text": chunk_text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.45,
                        "similarity_boost": 0.89
                    }
                }
                if previous_text:
                    payload["previous_text"] = previous_text
                if next_text:
                    payload["next_text"] = next_text

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_message = await response.text()
                        raise ValueError(f"Elevenlabs API error: {response.status} - {error_message}")
                    yield await response.read()
            except ValueError as ve:
                logger.error(f"Error in elevenlabs_generator: {str(ve)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in elevenlabs_generator: {str(e)}")
                raise ValueError(f"Unexpected error in TTS generation: {str(e)}")


async def openai_generator(voice_id: str, chunks: list):
    """Generate TTS audio using OpenAI API."""
    logger.info("Entering openai generator")
    url = "https://api.openai.com/v1/audio/speech"

    if not openai_key:
        raise ValueError("OpenAI API key not configured")

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        for chunk_text in chunks:
            try:
                payload = {
                    "model": "tts-1",
                    "input": chunk_text,
                    "voice": voice_id,
                    "response_format": "mp3"
                }

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_message = await response.text()
                        raise ValueError(f"OpenAI TTS API error: {response.status} - {error_message}")
                    yield await response.read()
            except ValueError as ve:
                logger.error(f"Error in openai_generator: {str(ve)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in openai_generator: {str(e)}")
                raise ValueError(f"Unexpected error in OpenAI TTS generation: {str(e)}")

async def handle_tts_request(websocket: WebSocket, data: dict, current_user: User, is_whatsapp=False, sample_voice_id=None):
    try:
        text = data.get('text', '')
        author = data.get('author', 'bot')
        conversationId = data.get('conversationId', None)
        if sample_voice_id:
            voice_id = sample_voice_id
        
        logger.debug(f"Received: text='{text}', author='{author}', conversationId={conversationId}, is_whatsapp={is_whatsapp}, sample_voice_id={sample_voice_id}")
        
        if conversationId is None and not voice_id:
            return None, "Conversation ID not found or not available"            
        
        if text:
            text = process_text_for_tts(text)
        
        try:
            if not sample_voice_id:
                characters_used = len(text)
                tts_cost = characters_used * Cost.TTS_COST_PER_CHARACTER
                
                if not await has_sufficient_balance(current_user.id, tts_cost):
                    if is_whatsapp:
                        return None, 'insufficient-balance'
                    if websocket:
                        await manager.send_json(websocket, {'action': 'insufficient-balance'})
                    return

                if author == 'user':
                    voice_id = current_user.voice_code if current_user.voice_code else "nMPrFLO7QElx9wTR0JGo"
                elif author == 'bot':
                    voice_id = await get_voice_code_from_conversation(conversationId, current_user)
                else:
                    # Use a default voice if the author is neither 'user' nor 'bot'
                    voice_id = "nMPrFLO7QElx9wTR0JGo"  
                    logger.debug(f"Warning: Unknown author '{author}'. Using default voice.")

                logger.debug(f"voice_id: {voice_id}")
                
        except ValueError as e:
            if is_whatsapp:
                return None, str(e)
            if websocket:
                await manager.send_json(websocket, {'action': 'error', 'message': str(e)})
            return

        logger.debug(f"Before hash_digest, voice_id: {voice_id}")

        hash_input = f"{text}_{voice_id}"
        hash_digest = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        file_path, full_path_opus = get_file_path(hash_digest)

        logger.debug(f"full_path_opus: {full_path_opus}")

        if os.path.exists(full_path_opus):
            logger.debug(f"Cached audio found at: {full_path_opus}")
            if is_whatsapp or not websocket:
                return full_path_opus, None
            else:
                await send_cached_audio(websocket, full_path_opus)
            return full_path_opus, None
        else:
            os.makedirs(file_path, exist_ok=True)

            if not sample_voice_id:
                chunks = await insert_tts_break(text)
                if not chunks:
                    error_message = "No audio content could be generated."
                    logger.error(error_message)
                    return None, error_message
            else:
                chunks = [text]
            
            logger.debug(f"Number of text chunks: {len(chunks)}")

            if not sample_voice_id:
                await cost_tts(current_user.id, tts_cost, characters_used)

            try:
                audio_generator = get_tts_generator(tts_engine, voice_id, chunks)
                audio_segments = []

                async for audio_chunk in audio_generator:
                    if websocket and not is_whatsapp:
                        await manager.send_bytes(websocket, audio_chunk)
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_chunk), format='mp3')
                    audio_segments.append(audio_segment)

                if audio_segments:
                    combined_audio = audio_segments[0]
                    for segment in audio_segments[1:]:
                        combined_audio += segment

                    combined_audio.export(full_path_opus, format="ogg", codec="libopus")
                    logger.debug(f"Audio opus cached at: {full_path_opus}")

                    if is_whatsapp or not websocket:
                        return full_path_opus, None
                    else:
                        await manager.send_json(websocket, {'action': 'finished'})
                        return full_path_opus, None
                else:
                    error_message = "No audio segments were generated."
                    logger.error(error_message)
                    return None, error_message
            except ValueError as ve:
                error_message = f"TTS generation failed: {str(ve)}"
                logger.error(error_message)
                return None, error_message
            except Exception as e:
                error_message = f"Unexpected error in TTS generation: {str(e)}"
                logger.error(error_message)
                return None, error_message

    except Exception as e:
        error_message = f"Error in handle_tts_request: {str(e)}"
        logger.error(error_message)
        return None, error_message        
    
async def handle_openai_tts_request(websocket: WebSocket, data: dict, current_user: User):
    text = data.get('text', '')
    voice = data.get('voice', 'onyx')  # Get voice from request, default to onyx
    logger.debug(f"OpenAI TTS voice: {voice}")

    characters_used = len(text)
    tts_cost = characters_used * Cost.TTS_COST_PER_CHARACTER
    logger.debug(f"TTS cost: {tts_cost}, cost per char: {Cost.TTS_COST_PER_CHARACTER}")

    if not await has_sufficient_balance(current_user.id, tts_cost):
        await websocket.send_text(orjson.dumps({'action': 'insufficient-balance'}).decode())
        return

    chunks = await insert_tts_break(text)
    if not chunks:
        await websocket.send_text(orjson.dumps({'action': 'no-content'}).decode())
        return

    await cost_tts(current_user.id, tts_cost, characters_used)

    # Include voice in hash for proper cache differentiation
    hash_input = f"{text}_{voice}"
    hash_digest = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    file_path, full_path_opus = get_file_path(hash_digest)

    if os.path.exists(full_path_opus):
        logger.debug(f"Cached file exists: {full_path_opus}")
        with open(full_path_opus, "rb") as f:
            while chunk := f.read(1024):
                await websocket.send_bytes(chunk)

        await websocket.send_text(orjson.dumps({'action': 'finished'}).decode())
    else:
        os.makedirs(file_path, exist_ok=True)

        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }

        audio_segments = []

        async with aiohttp.ClientSession() as session:
            for chunk_text in chunks:
                payload = {
                    "model": "tts-1",
                    "input": chunk_text,
                    "voice": voice,
                    "response_format": "opus"
                }

                async with session.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_message = await response.text()
                        logger.error(f"OpenAI TTS API error: {response.status} - {error_message}")
                        await websocket.send_text(orjson.dumps({'action': 'error', 'message': f'TTS error: {response.status}'}).decode())
                        return
                    audio_chunk = await response.read()
                    await websocket.send_bytes(audio_chunk)
                    audio_segment = AudioSegment.from_ogg(io.BytesIO(audio_chunk))
                    audio_segments.append(audio_segment)

        combined_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            combined_audio += segment

        combined_audio.export(full_path_opus, format="ogg", codec="libopus")
        logger.debug(f"Audio cached at: {full_path_opus}")
        await websocket.send_text(orjson.dumps({'action': 'finished'}).decode())
