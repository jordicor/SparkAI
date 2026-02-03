# rediscfg.py

# Modified to initialize Redis when the application starts and avoid the 2 second delay to connect to the pool


import os
import dramatiq
from datetime import timedelta
from redis import asyncio as aioredis
from dramatiq.brokers.redis import RedisBroker
from log_config import logger

class RedisManager:
    _instance = None
    _sync_pool = None
    _async_pool = None
    _sync_client = None
    _async_client = None
    _broker = None

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._sync_pool is None:
            self._sync_pool = aioredis.ConnectionPool(
                host=self.REDIS_HOST,
                port=self.REDIS_PORT,
                db=self.REDIS_DB,
                decode_responses=True,
                max_connections=30,
                health_check_interval=45,
                socket_keepalive=True,
                socket_timeout=300,
                retry_on_timeout=True
            )

        if self._async_pool is None:
            self._async_pool = aioredis.ConnectionPool(
                host=self.REDIS_HOST,
                port=self.REDIS_PORT,
                db=self.REDIS_DB,
                decode_responses=True,
                max_connections=30
            )

        if self._broker is None:
            self._broker = RedisBroker(host=self.REDIS_HOST, port=self.REDIS_PORT)
            # Add middleware
            self._broker.add_middleware(dramatiq.middleware.AgeLimit(max_age=300000))  # 5 minutes
            self._broker.add_middleware(dramatiq.middleware.TimeLimit(time_limit=600000))  # 10 minutes
            self._broker.add_middleware(dramatiq.middleware.Retries(max_retries=0))  # No retries
            dramatiq.set_broker(self._broker)

    def get_sync_client(self) -> aioredis.Redis:
        if self._sync_client is None:
            self._sync_client = aioredis.Redis(connection_pool=self._sync_pool)
        return self._sync_client

    def get_async_client(self) -> aioredis.Redis:
        if self._async_client is None:
            self._async_client = aioredis.Redis(connection_pool=self._async_pool)
        return self._async_client

    def get_broker(self) -> RedisBroker:
        return self._broker

    @classmethod
    async def close(cls):
        if cls._instance:
            if cls._instance._async_client:
                await cls._instance._async_client.close()
            if cls._instance._sync_client:
                await cls._instance._sync_client.close()
            if cls._instance._sync_pool:
                await cls._instance._sync_pool.disconnect()
            if cls._instance._async_pool:
                await cls._instance._async_pool.disconnect()
            if cls._instance._broker:
                cls._instance._broker.shutdown()
            cls._instance = None

# Get the manager instance
redis_manager = RedisManager.get_instance()

# Get the clients and broker
redis_client = redis_manager.get_async_client()
broker = redis_manager.get_broker()

async def add_revoked_user(user_id: int):
    try:
        # Add the user ID to Redis with a 4 hour expiration time
        await redis_client.setex(f"revoked_user:{user_id}", timedelta(hours=4), 1)
        return True
    except Exception as e:
        logger.error(f"Error adding revoked user to Redis: {e}")
        return False

async def is_user_revoked(user_id: int) -> bool:
    try:
        exists = await redis_client.exists(f"revoked_user:{user_id}")
        return bool(exists)
    except Exception as e:
        logger.error(f"Error checking revoked user in Redis: {e}")
        return False

async def close_redis_connection():
    await RedisManager.close()

# Rate limiting functions
async def check_rate_limit(user_id: int, action: str = "ai_call", limit: int = 30, window_minutes: int = 1) -> bool:
    """
    Check if user has exceeded rate limit for a specific action.
    Uses sliding window counter with Redis.
    
    Args:
        user_id: User ID to check
        action: Action type (default: 'ai_call')  
        limit: Max requests per window (default: 30)
        window_minutes: Time window in minutes (default: 1)
    
    Returns:
        True if under limit, False if exceeded
    """
    try:
        import time
        current_time = int(time.time())
        window_start = current_time - (window_minutes * 60)
        
        # Use sorted set to track requests in time window
        key = f"rate_limit:{action}:{user_id}"
        
        # Remove old entries outside the window
        await redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests in window
        current_count = await redis_client.zcard(key)
        
        if current_count >= limit:
            logger.warning(f"Rate limit exceeded for user {user_id}, action {action}: {current_count}/{limit}")
            return False
        
        # Add current request to set
        await redis_client.zadd(key, {str(current_time): current_time})
        
        # Set expiry for the key (cleanup old keys)
        await redis_client.expire(key, window_minutes * 60 + 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking rate limit for user {user_id}: {e}")
        # On Redis error, allow request (fail open)
        return True

async def get_rate_limit_status(user_id: int, action: str = "ai_call", limit: int = 30, window_minutes: int = 1) -> dict:
    """
    Get current rate limit status for user.
    
    Returns:
        Dict with current count, limit, and reset time
    """
    try:
        import time
        current_time = int(time.time())
        window_start = current_time - (window_minutes * 60)
        
        key = f"rate_limit:{action}:{user_id}"
        
        # Clean old entries and count current
        await redis_client.zremrangebyscore(key, 0, window_start)
        current_count = await redis_client.zcard(key)
        
        # Calculate when the oldest request will expire
        oldest_requests = await redis_client.zrange(key, 0, 0, withscores=True)
        reset_time = None
        if oldest_requests:
            oldest_timestamp = int(oldest_requests[0][1])
            reset_time = oldest_timestamp + (window_minutes * 60)
        
        return {
            "current": current_count,
            "limit": limit,
            "remaining": max(0, limit - current_count),
            "reset_time": reset_time,
            "window_minutes": window_minutes
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limit status for user {user_id}: {e}")
        return {
            "current": 0,
            "limit": limit,
            "remaining": limit,
            "reset_time": None,
            "window_minutes": window_minutes
        }

# Basic metrics functions
async def increment_metric(metric_name: str, value: int = 1, ttl_hours: int = 24):
    """
    Increment a metric counter in Redis.
    
    Args:
        metric_name: Name of the metric (e.g., 'api_calls', 'ai_requests', 'users_active')
        value: Value to increment by (default: 1)
        ttl_hours: Hours to keep the metric (default: 24)
    """
    try:
        key = f"metrics:{metric_name}"
        await redis_client.incrby(key, value)
        await redis_client.expire(key, ttl_hours * 3600)
    except Exception as e:
        logger.error(f"Error incrementing metric {metric_name}: {e}")

async def get_metrics() -> dict:
    """
    Get all current metrics from Redis.
    
    Returns:
        Dict with metric names and their current values
    """
    try:
        # Get all metric keys
        metric_keys = await redis_client.keys("metrics:*")
        metrics = {}
        
        if metric_keys:
            # Get all values in one call
            values = await redis_client.mget(metric_keys)
            
            for key, value in zip(metric_keys, values):
                # Remove 'metrics:' prefix from key name
                clean_key = key.replace("metrics:", "")
                metrics[clean_key] = int(value) if value else 0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {}

async def increment_user_activity(user_id: int):
    """
    Track active users for basic analytics.
    Uses a set to count unique active users per hour.
    """
    try:
        import time
        current_hour = int(time.time() // 3600)  # Current hour as timestamp
        key = f"metrics:active_users:{current_hour}"
        
        await redis_client.sadd(key, user_id)
        await redis_client.expire(key, 7200)  # Keep for 2 hours
        
    except Exception as e:
        logger.error(f"Error tracking user activity for user {user_id}: {e}")

async def get_active_users_count() -> int:
    """
    Get count of active users in current hour.
    """
    try:
        import time
        current_hour = int(time.time() // 3600)
        key = f"metrics:active_users:{current_hour}"
        
        count = await redis_client.scard(key)
        return count
        
    except Exception as e:
        logger.error(f"Error getting active users count: {e}")
        return 0

async def close_redis_connection():
    await RedisManager.close()

# Export broker and Redis client for use in other files
__all__ = ['broker', 'redis_client', 'add_revoked_user', 'is_user_revoked', 'close_redis_connection', 'RedisManager', 'check_rate_limit', 'get_rate_limit_status', 'increment_metric', 'get_metrics', 'increment_user_activity', 'get_active_users_count']