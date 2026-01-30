import torch.nn as nn

#!/usr/bin/env python

# Usage: torchrun --nnodes=1 --nproc_per_node=1 bug.py

import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import optimize
torch.manual_seed(0)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    idx = torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(-math.log(max_period) / half * idx)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Model(torch.nn.Module):
    def __init__(self, channels=320):
        super().__init__()
        self.channels = channels
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(channels, 32*channels),
            torch.nn.Linear(32*channels, 32*channels),
            torch.nn.Linear(32*channels, channels),
        )

    def forward(self, x, timesteps):
        t_emb = timestep_embedding(timesteps, self.channels)
        return self.layers(x) * t_emb

torch.distributed.init_process_group("nccl", rank=0, world_size=1)

x = torch.randn((320,), device="cuda")
t = torch.randint(0, 1000, (8,), device="cuda")
model = Model().cuda()
ddp_model = DDP(model, bucket_cap_mb=1)

ddp_model(x, t)

jit_model = optimize("inductor")(ddp_model)
jit_model(x, t)