# werkzeug_profile_app.py
from flask import Flask
import os
import time
from werkzeug.middleware.profiler import ProfilerMiddleware

app = Flask(__name__)
profile_dir = './profile'
os.makedirs(profile_dir, exist_ok=True)
app.wsgi_app = ProfilerMiddleware(app.wsgi_app, profile_dir=profile_dir)

@app.route("/a/b", methods=["GET"])
def route_a_b():
    time.sleep(1)
    return {"message": "message from a/b"}

if __name__ == "__main__":
    app.run()

# 注意每次请求都会是一个独立的文件, snakeviz 会新起一个服务
# pip install snakeviz
# snakeviz ./profile