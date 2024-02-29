# Flask examples

## `lifecycle_app.py`

深入了解可参考: [https://flask.palletsprojects.com/en/2.3.x/lifecycle/](https://flask.palletsprojects.com/en/2.3.x/lifecycle/)

本例展示了一个使用 `before_request` 和 `after_request` 的例子

## `profile_app.py`

使用 Python 的内置 cProfile 模块进行 profiling

## `werkzeug_profile_app.py`

详细内容参考 [https://srinaveendesu.medium.com/profiling-your-web-application-120d1e2602de](https://srinaveendesu.medium.com/profiling-your-web-application-120d1e2602de)

注意: `werkzeug.middleware.profiler.ProfilerMiddleware` 实际上也是基于 `cProfile`

**疑问**: 使用其他 web server 例如 gunicorn 启动的 Flask 应用怎么做 profile