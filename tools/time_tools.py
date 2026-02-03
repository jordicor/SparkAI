# tools/time_tools.py

import pytz
from datetime import datetime
from tools import register_tool, register_function_handler
import orjson

def get_time(timezone):
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz)
    return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

def get_time_difference(timezone1, timezone2):
    tz1 = pytz.timezone(timezone1)
    tz2 = pytz.timezone(timezone2)
    time1 = datetime.now(tz1)
    time2 = datetime.now(tz2)
    diff = time2 - time1
    hours, remainder = divmod(abs(diff.total_seconds()), 3600)
    minutes, _ = divmod(remainder, 60)
    sign = '-' if diff.total_seconds() < 0 else '+'
    return f"{sign}{int(hours):02d}:{int(minutes):02d}"

def convert_time(time, from_timezone, to_timezone):
    from_tz = pytz.timezone(from_timezone)
    to_tz = pytz.timezone(to_timezone)
    time_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    from_time = from_tz.localize(time_obj)
    to_time = from_time.astimezone(to_tz)
    return to_time.strftime("%Y-%m-%d %H:%M:%S %Z")

# Handlers for each function
async def handle_get_time(function_arguments, *args, **kwargs):
    timezone = function_arguments['timezone']
    result = get_time(timezone)
    content = f"Current time in {timezone}: {result}"
    yield f"data: {orjson.dumps({'content': content, 'save_to_db': True}).decode()}\n\n"

async def handle_get_time_difference(function_arguments, *args, **kwargs):
    timezone1 = function_arguments['timezone1']
    timezone2 = function_arguments['timezone2']
    result = get_time_difference(timezone1, timezone2)
    content = f"Time difference between {timezone1} and {timezone2}: {result}"
    yield f"data: {orjson.dumps({'content': content, 'save_to_db': True}).decode()}\n\n"

async def handle_convert_time(function_arguments, *args, **kwargs):
    time = function_arguments['time']
    from_timezone = function_arguments['from_timezone']
    to_timezone = function_arguments['to_timezone']
    result = convert_time(time, from_timezone, to_timezone)
    content = f"Converted time: {result}"
    yield f"data: {orjson.dumps({'content': content, 'save_to_db': True}).decode()}\n\n"

# Register the tools for the semantic router
register_tool({
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a specific timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to get the time for (e.g., 'America/New_York')"
                }
            },
            "required": ["timezone"]
        }
    }
})

register_tool({
    "type": "function",
    "function": {
        "name": "get_time_difference",
        "description": "Get the time difference between two timezones",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone1": {
                    "type": "string",
                    "description": "The first timezone"
                },
                "timezone2": {
                    "type": "string",
                    "description": "The second timezone"
                }
            },
            "required": ["timezone1", "timezone2"]
        }
    }
})

register_tool({
    "type": "function",
    "function": {
        "name": "convert_time",
        "description": "Convert a time from one timezone to another",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "The time to convert in format 'YYYY-MM-DD HH:MM:SS'"
                },
                "from_timezone": {
                    "type": "string",
                    "description": "The source timezone"
                },
                "to_timezone": {
                    "type": "string",
                    "description": "The destination timezone"
                }
            },
            "required": ["time", "from_timezone", "to_timezone"]
        }
    }
})

# Register the function handlers
register_function_handler("get_time", handle_get_time)
register_function_handler("get_time_difference", handle_get_time_difference)
register_function_handler("convert_time", handle_convert_time)