import torch.nn as nn

raise NotImplementedError("Only single local shard is supported.")

"""Minimal repro.

torchrun --nproc_per_node=2 fsdp_bug.py
"""
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class ToyParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.rand((1, 16)))

    def forward(self, x):
        return self.param * x


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    model = ToyParamModel().cuda()
    fsdp_model = FSDP(model)
    _ = get_model_state_dict(fsdp_model)


if __name__ == "__main__":
    main()