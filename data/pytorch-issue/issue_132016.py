# torch.rand(123, 88, dtype=torch.float32)
import os
import torch
from torch import nn
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor

class MyModel(nn.Module):
    def forward(self, x):
        return torch._foreach_mul([x], torch.tensor(2.0))[0]

def my_model_function():
    return MyModel()

def GetInput():
    # Set environment variables if not already set
    if not os.environ.get("RANK"):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12312"

    # Initialize process group if not already done
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1)

    mesh = init_device_mesh("cuda", (1,))
    big_tensor = torch.randn(123, 88).cuda()  # Matches input shape comment
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
    return my_dtensor

