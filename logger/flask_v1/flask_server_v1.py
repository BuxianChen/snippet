from flask import Flask, request, g
from flask_logger_v1 import logger
from flask_process_v1 import foo


app = Flask(__name__)
app.logger = logger

@app.route("/test", methods=["POST"])
def test():
    params = request.get_json()
    g.rid = params["rid"]
    return {"result": foo(params["a"], params["b"])}

app.run(
    host="0.0.0.0",
    port=8000,
)
