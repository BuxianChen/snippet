# 拦截 openai 接口的输入输出
from functools import wraps
from openai import OpenAI
import json
from datetime import datetime

fw = open("openai_call_log_2.txt", "w")

def openai_api_log_decorate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"OPENAI Inputs: {args}, {kwargs}")
        api_input = kwargs['options'].json_data
        start_time = str(datetime.now())

        result = func(*args, **kwargs)
        print(f"OPENAI Output: {result}")
        api_output = json.loads(result.model_dump_json())
        end_time = str(datetime.now())
        fw.write(
            json.dumps(
                {
                    "input": json.dumps(api_input, ensure_ascii=False),
                    "output": json.dumps(api_output, ensure_ascii=False),
                    "start_time": start_time,
                    "end_time": end_time
                },
                ensure_ascii=False
            ) + "\n"
        )
        return result
    return wrapper

OpenAI._request = openai_api_log_decorate(OpenAI._request)