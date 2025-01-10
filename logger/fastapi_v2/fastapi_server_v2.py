import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi_logger_v2 import request_id_var
from fastapi_process_v2 import foo

app = FastAPI()


class Params(BaseModel):
    rid: str
    a: int
    b: int

@app.post("/test")
async def test(params: Params):
    request_id_var.set(params.rid)
    return {"result": foo(params.a, params.b)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
