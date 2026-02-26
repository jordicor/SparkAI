import orjson
from database import get_db_connection
from auth import hash_password, verify_password, get_user_by_username, get_current_user, create_access_token, get_user_by_id, get_user_from_phone_number
from auth import get_current_user_from_websocket, get_user_id_from_conversation, get_user_by_token, create_user_info, create_login_response, generate_magic_link


async def is_whatsapp_conversation(conversation_id):
    async with get_db_connection(readonly=True) as conn:
        cursor = await conn.cursor()
        await cursor.execute('SELECT external_platforms FROM USER_DETAILS WHERE user_id IN (SELECT user_id FROM conversations WHERE id = ?)', (conversation_id,))
        result = await cursor.fetchone()
        if result:
            external_platforms = orjson.loads(result[0]) if result[0] else {}
            whatsapp_data = external_platforms.get('whatsapp', {})
            whatsapp_conversation_id = whatsapp_data.get('conversation_id')
            if whatsapp_conversation_id is None:
                return False
            try:
                return int(whatsapp_conversation_id) == int(conversation_id)
            except (TypeError, ValueError):
                return str(whatsapp_conversation_id) == str(conversation_id)
    return False
