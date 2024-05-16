from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.utils import Input, Output
from typing import Any, Dict, List, Tuple


class CustomRunnable(RunnableSerializable[Input, Output]):
    def invoke(self, input, config=None, **kwargs):
        assert "history" in input
        turn = input.get("metadata", {}).get("turn", 0) + 1
        return {
            "output": input["question"] + "-turn" + str(turn),
            "metadata": {
                "turn": turn
            }
        }

class Wrap(RunnableSerializable[Input, Output]):
    runnable: RunnableSerializable
    memory: Dict[str, Dict[str, Any]] = {}

    def invoke(self, input, config=None, **kwargs) -> Any:
        session_id = input.pop("session_id", None)
        if session_id not in self.memory:
            self.memory[session_id] = {"history": [], "metadata": {}}

        # 更新历史记录
        if "history" in input:
            self.memory[session_id]["history"] = input["history"]

        # 更新元数据
        if "metadata" in input:
            self.memory[session_id]["metadata"] = input["metadata"]

        # 将历史记录和元数据添加到输入中
        input["history"] = self.memory[session_id]["history"]
        input["metadata"] = self.memory[session_id]["metadata"]

        # 调用原始链
        output = self.runnable.invoke(input, config=config, **kwargs)

        # 更新历史记录和元数据
        self.memory[session_id]["history"] = input["history"] + [("user", input["question"]), ("ai", output["output"])]
        if "metadata" in output:
            self.memory[session_id]["metadata"] = output["metadata"]

        return output["output"]
    
if __name__ == "__main__":
    runnable = CustomRunnable()
    wrap_runnable = Wrap(runnable=runnable)

    output = runnable.invoke(
        {
            "history": [],
            "question": "my age in 21"
        }
    )
    print(output)

    turn_1_output: str = wrap_runnable.invoke(
        {
            "question": "yy",
            "session_id": "session-1"
        }
    )
    print(turn_1_output)
    print(wrap_runnable.memory)

    turn_2_output: str = wrap_runnable.invoke(
        {
            "question": "zz",
            "session_id": "session-1"
        }
    )

    print(turn_2_output)
    print(wrap_runnable.memory)

    another_output: str = wrap_runnable.invoke(
        {
            "question": "zz",
            "session_id": "session-2"
        }
    )

    print(another_output)
    print(wrap_runnable.memory)
