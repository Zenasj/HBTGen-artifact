import os
import torch
from torch import nn
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor

# torch.rand(123, 88, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, x):
        result_mul = torch._foreach_mul([x], 2.0)[0]
        try:
            result_div = torch._foreach_div([x], 2.0)[0]
            return torch.tensor(1.0)  # success
        except Exception:
            return torch.tensor(0.0)  # failure

def my_model_function():
    return MyModel()

def GetInput():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12312"
    torch.distributed.init_process_group(backend="nccl", world_size=1)

    mesh = init_device_mesh("cuda", (1,))
    tensor = torch.randn(123, 88)
    dtensor = distribute_tensor(tensor, mesh, [Shard(0)])
    return dtensor

