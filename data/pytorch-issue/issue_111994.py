from transformers import Conv1D
import torch.nn as nn
import torch
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor

mesh = DeviceMesh("cpu", list(range(2)))

def my_fn(gm, inputs):
    print(gm.graph)
    return gm

model = Conv1D(8, 16)

# parallelize with DTensor
model.weight = nn.Parameter(distribute_tensor(model.weight, mesh, [Shard(1)]))
model.bias = nn.Parameter(distribute_tensor(model.bias, mesh, [Shard(0)]))

input = distribute_tensor(torch.randn(4, 16), mesh, [Shard(1)])
torch.compile(model, backend=my_fn)(input)