from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if request.path in ['/a/b']:
        end_time = time.time()
        execution_time = end_time - request.start_time
        original_data = response.json
        original_data["execution_time"] = execution_time
        response.set_data(jsonify(original_data).data)
    return response

@app.route('/a/b', methods=["POST"])
def route_a_b():
    t1 = time.time()
    r = "This is route /a/b"
    t2 = time.time()
    return {"result": r, "inner_time": t2 -t1}

@app.route('/a/c')
def route_a_c():
    t1 = time.time()
    r = "This is route /a/c"
    t2 = time.time()
    return {"result": r, "inner_time": t2 -t1}

if __name__ == '__main__':
    app.run()