# automatic repro does not work
#DOESNT CRASH WITH BACKEND='EAGER', unless torchdynamo_repro_after is uncommented
#Unsupported: meta converter nyi with fake tensor propagation.

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn import Conv2d
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.nn.utils import weight_norm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_COMPILE_DEBUG'] = '1'
    #os.environ['TORCHDYNAMO_REPRO_AFTER']="dynamo" #uncomment for a different failure mode :)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2d(1, 128, (5,1))
        self.c2 = weight_norm(Conv2d(128, 512, (5, 1)))
    def forward(self, x):
        a = self.c2(self.c1(x))
        c = self.c2(self.c1(x))
        return a, c

def demo_basic(rank, world_size):
    print(f"Running on rank {rank}.")
    setup(rank, world_size)
    model = Network()
    model = torch.compile(model)
    ddp_model = DDP(model)
    outputs = ddp_model(torch.randn(20, 1, 16, 16))

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size=2
    mp.spawn(demo_basic,
             args=(world_size,),
             nprocs=world_size)