#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    os.environ['MASTER_ADDR'] = 'a.b.c.d'
    os.environ['MASTER_PORT'] = '29500'
    print(f"Starting initializing {args.rank} world size is {args.world_size}")
    
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    print(f"Done initializing {args.rank}")

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()

def run(args):
    """ Distributed function to be implemented later. """
    # print(f"Rank is {rank} and size is {size}")
    # pass
    init_distributed_mode(args)
    fix_random_seeds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--batch_size", type=int, default=360)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--dist_url", type=str)
    args = parser.parse_args()
    run(args)