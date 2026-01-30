# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import time
import torch
import torch.distributed as dist
import datetime


def run_worker(rank, world_size):
    ndev = torch.cuda.device_count()
    device = torch.device(f'cuda:{rank % ndev}')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=10)
                           )

    numel = 100 * 1024 * 1024
    if rank == 0:
        numel *= 2
        print(f'Rank {rank} is the problematic rank')

    tensor = torch.ones(numel, device=device)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    print(f"Rank {rank} completed\n")


if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", 8))
    rank = int(os.getenv("RANK", -1))
    run_worker(rank, world_size)