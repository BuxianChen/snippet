# `openai-python` examples

**`catcher.py`**

`catcher.py` 中包含对所有使用 `openai-python` 包对 openai 模型调用输入输出进行拦截的工具 (拦截范围包含 GPT 系列, Embedding 系列等所有的 API 接口), 以便于了解模型的输入输出. 由于 `langchain`, `llama_index` 等库底层都是使用 `openai-python` 来对 openai 模型进行调用, 因此都可以做到拦截. 而如果不使用 `openai-python` 而是直接发请求, 这种情况无法拦截. 注意 `langchain` 和 `llama_index` 还有许多三方集成工具能将过程可视化 