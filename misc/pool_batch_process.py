"""
维护一个池子, 每次池子满了就排序按 batch 处理一部分数据

用于这种情形: 假设要对大量文本进行bert模型分类, 但句子的长度分布迥异,
如果短句子和长句子在一个batch里会显著降低推理速度, 但对数据集整体排序又代价过高,
因此可以维护一个合适大小的池子, 池子满了就按排序处理一批
"""

import random

# N = 0
def process_batch(batch):
    # global N
    # N += len(batch)
    print(sum([len(text) for text in batch]) / len(batch))

def process_pool(pool, pool_num, batch_size, process_all=False):
    if process_all:
        num_batch = (pool_num - 1) // batch_size + 1
    else:
        num_batch = pool_num // batch_size
    pool = sorted(pool, key=lambda x: len(x))
    for batch_idx in range(num_batch):
        process_batch(pool[batch_idx*batch_size: (batch_idx+1)*batch_size])
    
    pool = pool[num_batch*batch_size:]
    pool_num = len(pool)
    return pool, pool_num


if __name__ == "__main__":
    texts = ["".join(random.choices(["a", "b", "c"], k=random.randint(10, 200))) for i in range(10000)]

    pool_size = 1000
    batch_size = 32

    pool = []
    pool_num = 0
    for text in texts:
        pool.append(text)
        pool_num += 1
        if pool_num == pool_size:
            pool, pool_num = process_pool(pool, pool_num, batch_size, process_all=False)
    if pool:
        pool, pool_num = process_pool(pool, pool_num, batch_size, process_all=True)