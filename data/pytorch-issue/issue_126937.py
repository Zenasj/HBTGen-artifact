import logging
import os
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import pdb
import sys

class MyModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, device=device), nn.Linear(4, 4, device=device)])
        self.output = nn.Linear(4, 4, device=device)
    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)
        return self.output(z)

def main():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    rank = dist.get_rank()
    model = MyModel(torch.cuda.current_device())
    policy = ModuleWrapPolicy({nn.ModuleList}) # fails because nn.ModuleList does not have forward
    # policy = ModuleWrapPolicy({nn.Linear}) # success since nn.Linear have forward
    fsdp_model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=policy,
    )
    if rank == 0:
        print(model)
    torch.manual_seed(dist.get_rank() + 1)
    x = torch.randn((4, 4), device="cuda")
    fsdp_model(x)


if __name__ == "__main__":
    main()