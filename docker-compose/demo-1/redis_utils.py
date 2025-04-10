from typing import Any, Optional
import redis.asyncio as redis
from dataclasses import dataclass
import json

@dataclass
class RedisConfig:
    host: str = "localhost"  # Changed from "redis" to "localhost" for local development
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

class RedisManager:
    """
    async with RedisManager(RedisConfig()) as redis_manager:
        await redis_manager.set_key("test", "value")
        # Connection automatically closes after the with block
    """

    def __init__(self, config: RedisConfig):
        self.config = config
        self.client = None
    
    async def connect(self) -> None:
        """Establish connection to Redis"""
        try:
            self.client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=True
            )
            # Test the connection
            await self.client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    async def close(self) -> None:
        """Close the Redis connection"""
        if self.client:
            await self.client.close()
            self.client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def set_key(self, key: str, value: Any, expiry: Optional[int] = None) -> None:
        """Set a key-value pair with optional expiry in seconds"""
        if not self.client:
            await self.connect()
        if not isinstance(value, str):
            value = json.dumps(value)
        await self.client.set(key, value, ex=expiry)
    
    async def get_key(self, key: str) -> Any:
        """Get value for a key"""
        if not self.client:
            await self.connect()
        value = await self.client.get(key)
        try:
            return json.loads(value) if value else None
        except json.JSONDecodeError:
            return value

if __name__ == "__main__":
    RedisConfig()
    redis_manager  = RedisManager()

