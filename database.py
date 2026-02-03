# database.py
import aiosqlite
from contextlib import asynccontextmanager
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
        
dbname = os.getenv('DATABASE')
DEFAULT_DB_TIMEOUT = float(os.getenv('DB_TIMEOUT', '5.0'))
DB_MAX_RETRIES = int(os.getenv('DB_MAX_RETRIES', '3'))
DB_RETRY_DELAY_BASE = float(os.getenv('DB_RETRY_DELAY_BASE', '0.2'))

LOCK_ERROR_MESSAGES = ("database is locked", "database table is locked")

CACHE_SIZE = 1024 * 50  # 50 MB cache
PRAGMA_STATEMENTS_RW = [
    "PRAGMA journal_mode=WAL",
    f"PRAGMA cache_size={-CACHE_SIZE}",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA mmap_size=1073741824",
    "PRAGMA busy_timeout=5000",
]

PRAGMA_STATEMENTS_RO = [
    f"PRAGMA cache_size={-CACHE_SIZE}",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA mmap_size=1073741824",
    "PRAGMA busy_timeout=5000",
]


@asynccontextmanager
async def get_db_connection(readonly=False):
    mode = 'ro' if readonly else 'rw'
    conn = await aiosqlite.connect(
        f"file:data/{dbname}?mode={mode}",
        uri=True,
        timeout=DEFAULT_DB_TIMEOUT,
    )
    conn.row_factory = aiosqlite.Row
    
    # Apply PRAGMA configurations
    pragma_statements = PRAGMA_STATEMENTS_RO if readonly else PRAGMA_STATEMENTS_RW
    for statement in pragma_statements:
        await conn.execute(statement)
    
    try:
        yield conn
    finally:
        await conn.close()


def is_lock_error(exc: Exception) -> bool:
    """
    Detects if the exception corresponds to a SQLite lock.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return any(token in message for token in LOCK_ERROR_MESSAGES)
