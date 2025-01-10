from flask_logger_v2 import logger

def foo(a, b):
    logger.info(f"a:{a},b:{b}")
    return a + b