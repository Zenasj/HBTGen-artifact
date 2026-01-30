#!/usr/bin/env python
import os
import time
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.distributed.rpc as rpc

def run(rank, size):
    if rank == 0:
        time.sleep(0.5) # to allow the other process to exit without joining
        ret = rpc.rpc_async("worker1", torch.add, args=(torch.ones(2), 2))
        result = ret.wait()

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29509'
    dist.init_process_group(backend, rank=rank, world_size=size, timeout=timedelta(seconds=12))
    rpc.init_model_parallel("worker{}".format(rank))
    fn(rank, size)
    if rank == 0: rpc.join_rpc()


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()