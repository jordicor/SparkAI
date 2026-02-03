# tasks.py

from rediscfg import broker, redis_client
import dramatiq
import asyncio

# Import functions to generate and save PDF and MP3
#from tools import *
from tools import dramatiq_tasks

#from tools.perplexity import query_perplexity
from tools.download_pdf import generate_and_save_pdf
from tools.download_mp3 import generate_and_save_mp3

# Define task to generate PDFs
@dramatiq.actor
def generate_pdf_task(conversation_id: int, user_id: int, is_admin: bool):
    import asyncio
    asyncio.run(generate_and_save_pdf(conversation_id, user_id, is_admin))

# Define task to generate MP3s
@dramatiq.actor
def generate_mp3_task(conversation_id: int, user_id: int, is_admin: bool):
    import asyncio
    asyncio.run(generate_and_save_mp3(conversation_id, user_id, is_admin))


@dramatiq.actor
def download_elevenlabs_audio_task(conversation_id: int, session_id: str, user_id: int):
    import asyncio
    from elevenlabs_service import service as elevenlabs_service
    asyncio.run(elevenlabs_service.download_conversation_audio(conversation_id, session_id, user_id))

