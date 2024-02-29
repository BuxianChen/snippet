# profile_app.py
import cProfile
from flask import Flask
import time

app = Flask(__name__)

@app.route("/a/b", methods=["GET"])
def route_a_b():
    time.sleep(1)
    return {"message": "message from a/b"}

if __name__ == "__main__":
    # 使用如下两种方式之一进行 profile, 但都得使用 Ctrl + C 将服务关停才能获取到 profile_stats 文件然后可视化
    # 方法一: python profile_app.py
    cProfile.run("app.run()", "profile_stats")

    # 方法二:
    # app.run()
    # python -m cProfile -o profile_stats profile_app.py

# 使用以下方法对 profile 结果可视化
# import pstats
# p = pstats.Stats("profile_stats")
# p.sort_stats("cumulative").print_stats()