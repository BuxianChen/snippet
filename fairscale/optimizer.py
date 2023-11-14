import torch
import copy
from typing import List
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(28*28, 512)
        self.linear_2 = nn.Linear(512, 512)
        self.linear_3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        return x

# 完全参考 fairscale.optim.OSS 源代码
class OSS(torch.optim.Optimizer):
    def __init__(self, params, optim, **default):
        super().__init__(params, default)
        # 这里因为只想探讨 partition_parameters, 所以直接写死
        self.world_size = 2
        self._partition_parameters = []

    # 完全参考源代码
    def partition_parameters(self) -> List[List[dict]]:
        """Partitions parameters across distributed data parallel ranks.

        Returns a list of param_groups (which is a list of dict) where each
        element of the list contains the param_groups for a rank. Element 0
        corresponds to rank 0, etc. We need all the ranks for the broadcast
        inside step().
        """
        if len(self._partition_parameters) == 0:
            self._partition_parameters = [list() for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param_group in self.param_groups:
                param_lists: List[List] = [list() for _ in range(self.world_size)]
                for param in param_group["params"]:
                    # Add this param to rank with smallest size.
                    rank = sizes.index(min(sizes))
                    param_lists[rank].append(param)

                    # We're partitioning the optimizer state,
                    # so trainable parameters are the ones which really count
                    if param.requires_grad:
                        sizes[rank] += param.numel()
                    else:
                        # Spread frozen params on a per-tensor basis
                        # Mostly useful for balance partitions for fine tuning for instance
                        # Not required strictly speaking
                        sizes[rank] += 1

                for rank, params in enumerate(param_lists):
                    param_group_rank = copy.copy(param_group)
                    param_group_rank["params"] = params
                    self._partition_parameters[rank].append(param_group_rank)

        return self._partition_parameters

model = NeuralNetwork().to(0)
params = [
    {"params": model.linear_1.parameters(), "lr": 0.1},
    {"params": model.linear_2.parameters(), "lr": 0.2},
    {"params": model.linear_3.parameters(), "lr": 0.3},
]
optimizer = OSS(params=params, optim=torch.optim.SGD, lr=0.001)

print(optimizer.partition_parameters())