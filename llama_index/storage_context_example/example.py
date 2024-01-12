import os
os.environ["OPENAI_API_KEY"] = "sk-xxx"  # changed to your api key
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader("./txt_data").load_data()
from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)  # 这里调用了一次 embedding 接口, 对两个 chunk 进行了 embedding

print(index.storage_context.to_dict())