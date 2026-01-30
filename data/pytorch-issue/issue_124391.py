import torch.nn as nn

import os
import torch

import torch._inductor.config as tic 
import torch.distributed as dist

RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

torch.cuda.set_device(RANK)

process_group = torch.distributed.init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=RANK,
)

def func(x):
    y = x * x

    if WORLD_SIZE > 1:
        y = dist.all_reduce(y, op=dist.ReduceOp.SUM, group=process_group)

    x = torch.nn.functional.silu(x)
    return x * y

tic.fx_graph_cache = False

options = {
    'triton.cudagraphs': True,
    'triton.cudagraph_trees': True, # Toggle this to True to enable/disable cudagraph trees
}

with torch.no_grad():
    func = torch.compile(func, backend='inductor', fullgraph=True, options=options, dynamic=None)

    for nelem in [1024, 2048, 4096]:
        x = torch.randn(nelem, device='cuda', dtype=torch.bfloat16)

        for _ in range(3):
            y = func(x)