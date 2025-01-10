from flask import current_app

def foo(a, b):
    current_app.logger.info(f"a:{a},b:{b}")
    return a + b