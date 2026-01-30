import torch.nn as nn

import os
import torch
from torch.distributed import fsdp
import torch.multiprocessing as mp

def run(rank, world_size):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        world_size=world_size,
        rank=rank,
    )

    ffn = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10))

    ffn = fsdp.FullyShardedDataParallel(ffn, device_id=rank)

    x1 = torch.rand((10, 10)).cuda()
    loss1 = ffn(x1).sum()

    x2 = torch.rand((10, 10)).cuda()
    loss2 = ffn(x2).sum()

    loss1.backward()

    #ffn._handle._needs_pre_backward_unshard = True
    loss2.backward()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = 2
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )