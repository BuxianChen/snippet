# 最简理解 StorageContext

数据源包含一个 TXT 文件, 但文件内容稍长, 需要切为两个 chunk 做 embedding, 文件内容来源 (TXT 文件 内容直接截取自下面这个文档):

```python
from llama_index import download_loader
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)
```

运行

```bash
python example.py
```

运行后输出简化为 `storage_context.json`, 简化的部分如下:

- emdedding 的结果原本是 1536 个浮点数, 简化为 3 个浮点数
- 原始文本被简化为了: `paragraph chunk 1 ...`
- 省略了一些 metadata 的键值对
- 一些 Enum 类型, 使用类似 `"<ObjectType.TEXT: 1>"` 来代替


**解读**

`storage_context` 包含如下的 `Store`:

- `vector_store.default[.embedding_dict]` 中存储 embedding 结果
- `doc_store.[docstore/data]` 中存储 Node 信息, 包含文本内容, Node 的关系 (parent, next, previous)
- `index_store`: 应该只存一些 id 映射信息
- `graph_store`: 此例不涉及
- `vector_store.image`: 此例不涉及

注意默认的切分方式里, 两个 chunk 实际上是有重叠的, 并且送入 Embedding 模型的数据实际上还带了前缀:

```python
prefix = "file_path: txt_data/galaxy.txt"
chunks = [
    {
        "text": "pragraph chunk 1 ...",
        "start_char_idx": 0,
        "end_char_idx": 4143,
    },
    {
        "text": "pragraph chunk 2 ...",
        "start_char_idx": 3281,
        "end_char_idx": 4491,
    }
]
```
