import time
from datetime import timedelta

import torch
import torch.cuda
import torch.distributed
import torch.multiprocessing
import torch.nn
import torch.optim
import torch.utils.data

world_size = 4
master_rank = 0


def init_distributed(
    rank: int,
    world_size: int = world_size,
    backend: str = "nccl",
    hostname: str = "127.0.0.1",
    port: int = 29500,
    timeout: timedelta = timedelta(minutes=30),
) -> torch.distributed.TCPStore:
    is_master = rank == master_rank
    store = torch.distributed.TCPStore(hostname, port, world_size, is_master, timeout)
    torch.distributed.init_process_group(
        backend, store=store, timeout=timeout, world_size=world_size, rank=rank
    )
    print("Worker %d started." % rank)
    return store


def run_worker(rank):
    store = init_distributed(rank)
    if rank == master_rank:
        keys = [str(i) for i in range(world_size) if i != master_rank]
        time.sleep(2)
        print("Wait:", keys)
        store.wait(keys, timedelta(seconds=world_size + 1))
    else:
        print("Sleep:", rank)
        time.sleep(rank)
        print("Set:", rank)
        store.set(str(rank), str(rank))


if __name__ == "__main__":
    torch.multiprocessing.spawn(run_worker, nprocs=world_size, args=tuple())