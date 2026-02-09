# tools/perplexity.py

import os
import orjson
import asyncio
import aiohttp
import dramatiq
import time
from dotenv import load_dotenv
from rediscfg import redis_client, broker
from tools import register_tool, register_dramatiq_task, register_function_handler

load_dotenv()

PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

async def query_perplexity(query, publish_func):
    print(f"enters query_perplexity, query: {query}")
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "messages": [
            {
                "content": "Answer the user's question in the same language they use. Use Markdown format when possible. Prioritize reliable and up-to-date sources, including citations or references. Provide a concise summary followed by relevant details.",
                "role": "system"
            },
            {
                "content": query,
                "role": "user"
            }
        ],
        "model": "sonar-pro",
        "stream": True,
        "return_citations": True,
        "return_images": False,
        "return_related_questions": False,
        "temperature": 0.9
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            async for line in response.content:
                if line:
                    try:
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith('data: '):
                            data = orjson.loads(line_text[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    await publish_func(delta['content'])
                    except Exception as e:
                        print(f"Error in query_perplexity: {e}")

@dramatiq.actor
def query_perplexity_task(channel_name: str, query: str):
    asyncio.run(perform_perplexity_query(channel_name, query))

async def perform_perplexity_query(channel_name: str, query: str):
    print(f"Processing query: {query}")
    content = ""

    async def publish_func(message):
        await redis_client.publish(channel_name, message)

    try:
        await query_perplexity(query, publish_func)
    except Exception as e:
        print(f"Error in perform_perplexity_query: {e}")
        await publish_func(orjson.dumps({'error': str(e)}).decode())
    finally:
        await publish_func('END')

async def handle_perplexity_query(function_arguments, messages, model, temperature, max_tokens, content, conversation_id, current_user, request, input_tokens, output_tokens, total_tokens, message_id, user_id, client, prompt, user_message=None):
    query = function_arguments['query']
    channel_name = f"perplexity_response_{conversation_id}_{user_id}_{int(time.time())}"

    # Launch the task
    query_perplexity_task.send(channel_name, query)

    # Subscribe to the channel and yield the messages
    content = ""
    pubsub = None
    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel_name)

        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0)
            if message:
                data = message['data']
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                if data == 'END':
                    break
                content += data
                yield f"data: {orjson.dumps({'content': data}).decode()}\n\n"
            else:
                await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error in handle_perplexity_query: {e}")
        yield f"data: {orjson.dumps({'error': str(e)}).decode()}\n\n"
    finally:
        if pubsub:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()

    # Estimate tokens (rough: 4 chars = 1 token)
    input_tokens_estimate = len(query) // 4
    output_tokens_estimate = len(content) // 4

    # Perplexity sonar-pro pricing (per 1M tokens)
    PERPLEXITY_INPUT_COST = 3.0 / 1_000_000   # $3 per 1M
    PERPLEXITY_OUTPUT_COST = 15.0 / 1_000_000  # $15 per 1M

    perplexity_cost = (input_tokens_estimate * PERPLEXITY_INPUT_COST) + (output_tokens_estimate * PERPLEXITY_OUTPUT_COST)

    # Yield the final content with save_to_db flag and cost tracking
    final_data = {
        'content': content.strip(),
        'save_to_db': True,
        'perplexity_cost': perplexity_cost,
        'perplexity_input_tokens': input_tokens_estimate,
        'perplexity_output_tokens': output_tokens_estimate
    }
    yield f"data: {orjson.dumps(final_data).decode()}\n\n"

# Register the tool for the semantic router
register_tool({
    "type": "function",
    "function": {
        "name": "query_perplexity",
        "description": "Use this tool for up-to-date or real-time internet searches. Formulate a detailed natural language query for best results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Detailed natural language query for semantic search. Clearly specify what information you're looking for and what you expect to obtain."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },
    "strict": True
})

# Register the task for Dramatiq    
register_dramatiq_task("query_perplexity_task", query_perplexity_task)

# Register the function handler
register_function_handler("query_perplexity", handle_perplexity_query)