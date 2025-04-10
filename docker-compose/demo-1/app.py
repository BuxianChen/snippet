from redis_utils import RedisManager, RedisConfig
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

redis_config = RedisConfig(
    host=os.getenv("REDIS_URL", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379))
)
redis_manager = RedisManager(redis_config)

app = FastAPI()

class SetModel(BaseModel):
    key: str
    value: str

class GetModel(BaseModel):
    key: str


@app.post("/set")
async def set_key(input: SetModel):
    await redis_manager.set_key(key=input.key, value=input.value)
    return {
        "status": "ok",
        "key": input.key,
        "value": input.value
    }

@app.post("/get")
async def get_key(input: GetModel):
    value = await redis_manager.get_key(key=input.key)
    return {
        "status": "ok",
        "key": input.key,
        "value": value
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)