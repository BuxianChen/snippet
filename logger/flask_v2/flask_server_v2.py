from flask import Flask, request
from flask_logger_v2 import request_id_var
from flask_process_v2 import foo

app = Flask(__name__)

@app.before_request
def set_request_id():
    rid = request.json.get("rid")
    request_id_var.set(rid)

@app.route("/test", methods=["POST"])
def test():
    params = request.json
    return {"result": foo(params["a"], params["b"])}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
