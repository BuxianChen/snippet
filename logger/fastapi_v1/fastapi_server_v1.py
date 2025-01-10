import uvicorn
from fastapi import FastAPI, Depends, Request, HTTPException
import logging
from pydantic import BaseModel
from fastapi_logger_v1 import logger, RequestIdAdapter
from fastapi_process_v1 import foo

class Params(BaseModel):
    rid: str
    a: int
    b: int


app = FastAPI()

async def get_rid(request: Request):
    try:
        body = await request.json()
        rid = body.get("rid")
        if not rid:
            raise HTTPException(status_code=400, detail="rid is required in the request body")
        return rid
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body")

def get_logger(rid: str = Depends(get_rid)):
    return RequestIdAdapter(logger, {"rid": rid})

@app.post("/test")
async def test(params: Params, logger: logging.LoggerAdapter = Depends(get_logger)):
    return {"result": foo(params.a, params.b, logger)}

uvicorn.run(app, host="0.0.0.0", port=8000)