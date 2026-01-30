import torch
import torch.nn as nn
from torch.distributed.tensor import (
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
    init_device_mesh,
)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input) + self.fc2(input))


mesh = init_device_mesh("cuda", (4,))


def shard_params(mod_name, mod, mesh):
    col_linear_placement = [Shard(0)]
    # shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, col_linear_placement)
            )
            mod.register_parameter(name, dist_param)


sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)

x = torch.randn(8, 8).cuda(0)
x = distribute_tensor(x, mesh, [Replicate()])
y = sharded_module(x)
print(y.shape)