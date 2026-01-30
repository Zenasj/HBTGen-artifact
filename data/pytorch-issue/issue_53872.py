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
    for i in range(5):
        if rank == master_rank:
            keys = [str(i) for i in range(world_size) if i != master_rank]

            start = time.time()
            [store.wait([k]) for k in keys]
            print("`wait` ran for", (time.time() - start) * 1000, "milliseconds")

            start = time.time()
            [store.get(k) for k in keys]
            print("`get` ran for", (time.time() - start) * 1000, "milliseconds")

            start = time.time()
            [store.delete_key(k) for k in keys]
            print("`delete_key` ran for", (time.time() - start) * 1000, "milliseconds")
        else:
            start = time.time()
            store.set(str(rank), str(rank))
            print("[%s] `set` ran for" % rank, (time.time() - start) * 1000, "milliseconds")
        time.sleep(1)


if __name__ == "__main__":
    torch.multiprocessing.spawn(run_worker, nprocs=world_size, args=tuple())