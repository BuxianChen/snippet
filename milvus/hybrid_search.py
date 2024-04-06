import random
import string
import numpy as np

from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)
import random
from scipy.sparse import csr_array, vstack

dense_dim = 20
sparse_dim = 100
n = 10
docs = [f"text-{i}" for i in range(n)]
dense_vectors = [np.random.random(dense_dim).astype(np.float32) for i in range(n)]

sparse_vectors = []
for i in range(10):
    k = random.randint(3, 10)
    indices = [random.randint(0, sparse_dim - 1) for _ in range(k)]
    row_indices = [0] * k
    values = [random.random() for _ in range(k)]

    csr = csr_array((values, (row_indices, indices)), shape=(1, sparse_dim))
    sparse_vectors.append(csr)
sparse_vectors = vstack(sparse_vectors)

connections.connect("default", host="localhost", port="19530")

# Specify the data schema for the new Collection.
fields = [
    # Use auto generated id as primary key
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    # Milvus now supports both sparse and dense vectors, we can store each in
    # a separate field to conduct hybrid search on both vectors.
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]

schema = CollectionSchema(fields, "")
col_name = 'hybrid_demo'
# Now we can create the new collection with above name and schema.
col = Collection(col_name, schema, consistency_level="Strong")

# We need to create indices for the vector fields. The indices will be loaded
# into memory for efficient search.
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "FLAT", "metric_type": "L2"}
col.create_index("dense_vector", dense_index)
col.load()

# 3. insert text and sparse/dense vector representations into the collection
entities = [docs, sparse_vectors, dense_vectors]
col.insert(entities)

col.flush()


k = random.randint(3, 10)
indices = [random.randint(0, sparse_dim - 1) for _ in range(k)]
row_indices = [0] * k
values = [random.random() for _ in range(k)]
csr = csr_array((values, (row_indices, indices)), shape=(1, sparse_dim))

sparse_search_params = {"metric_type": "IP"}
sparse_req = AnnSearchRequest(
    data=csr,
    anns_field="sparse_vector",
    param=sparse_search_params,
    limit=3,
)
dense_search_params = {"metric_type": "L2"}
dense_req = AnnSearchRequest(
    data=dense_vectors[:1],
    anns_field="dense_vector",
    param=dense_search_params,
    limit=3
)

# 混合检索只支持 1 个向量
res = col.hybrid_search(
    reqs=[sparse_req, dense_req],
    rerank=RRFRanker(k=60),
    limit=3,
    output_fields=['text']
)

res = col.search(
    dense_vectors, "dense_vector", {"metric_type": "L2"}, limit=3, output_fields=["pk"]
)

res = col.search(
    csr, "sparse_vector", {"metric_type": "IP"}, limit=3, output_fields=["pk"]
)

res = col.search(
    sparse_vectors, "sparse_vector", {"metric_type": "IP"}, limit=3, output_fields=["pk"]
)

col.drop()