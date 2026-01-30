py
import os
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


policy = ModuleWrapPolicy({torch.nn.Linear})
mesh_2d = init_device_mesh("cuda", (2, 4))
model = FSDP(
    ToyModel(), 
    device_mesh=mesh_2d,
    # Either not passing sharding strategy or not passing autowrap policy fixes the issue
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    auto_wrap_policy=policy,
    device_id=int(os.environ["LOCAL_RANK"]),
)