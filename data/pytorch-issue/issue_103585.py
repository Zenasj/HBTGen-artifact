import torch.nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import transformers
import multiprocessing as mp
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    gptj = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    module = DistributedDataParallel(gptj)

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    world_size = 2
    run_demo(demo_basic, world_size)