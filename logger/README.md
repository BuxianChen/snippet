问题: 假设我希望使用 Flask/FastAPI 部署一个服务, 我希望为同一次请求的过程日志都加上 `rid:xxx` 的前缀, 以利于追踪每次请求的执行情况, 应该怎么做? 假设接口的实际处理逻辑发生在另一个代码逻辑内, 且另一个代码逻辑里会产生日志.

我们在不涉及部署 Flask/FastAPI 时, 纯处理逻辑的代码可能是这样的

在一个文件 `customer_logger.py` 中自定义 logger 的格式

```python
# customer_logger.py
import logging

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = get_logger()
```

在另一个文件 `process.py` 里实现业务逻辑

```python
# process.py
from customer_logger import logger   # 全局变量
def foo(a, b):
    logger.info(f"a:{a},b:{b}")
    return a + b
```

现在我们采用 `flask_logger.py` 需要将业务逻辑代码修改为:

```python
from flask import current_app
def foo(a, b):
    current_app.logger.info(f"a:{a},b:{b}")
    return a + b
```

而采用 `fastapi_logger.py` 需要将业务逻辑代码修改为:

```python
def foo(a, b, logger):
    # logger 需要从别处传入
    logger.info(f"a:{a},b:{b}")
```

最后, 我们用 `fastapi_logger_v2.py` 或 `flask_logger_v2.py`, 则业务代码可以原封不动
