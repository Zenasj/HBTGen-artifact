import os
import torch
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12312"
torch.distributed.init_process_group(backend="nccl", world_size=1)

mesh = init_device_mesh("cuda", (int(1),))
big_tensor = torch.randn(123, 88)
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])

torch._foreach_mul([my_dtensor], 2.0)  # ok
torch._foreach_div([my_dtensor], 2.0)  # not ok