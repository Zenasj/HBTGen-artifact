import torch.nn as nn

py
import torch
import os

from torch.distributed.tensor.parallel import ColwiseParallel

class Dummy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.x = torch.nn.Linear(1000, 1000)

    def forward(self, x):
        return self.y(self.x(x))

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

dummy = Dummy().to(device)

tp_plan = {"x": ColwiseParallel()}
device_mesh = torch.distributed.init_device_mesh("cuda", (world_size,))

torch.distributed.barrier()
torch.distributed.tensor.parallel.parallelize_module(
    dummy,
    device_mesh=device_mesh,
    parallelize_plan=tp_plan,
)

torch.distributed.barrier()
torch.distributed.destroy_process_group()

import torch
import os

from torch.distributed.tensor.parallel import ColwiseParallel

class Dummy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.x = torch.nn.Linear(1000, 1000)

    def forward(self, x):
        return self.y(self.x(x))

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
torch.distributed.init_process_group("nccl", device_id=device)

dummy = Dummy().to(device)

tp_plan = {"x": ColwiseParallel()}
device_mesh = torch.distributed.init_device_mesh("cuda", (world_size,))

torch.distributed.barrier()
torch.distributed.tensor.parallel.parallelize_module(
    dummy,
    device_mesh=device_mesh,
    parallelize_plan=tp_plan,
)
print(f"rank: {rank}, dummy: {dummy}")

torch.distributed.barrier()
torch.distributed.destroy_process_group()