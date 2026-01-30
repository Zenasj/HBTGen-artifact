import os
import time
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.distributed.rpc as rpc

def run(rank, size):
    if rank == 1:
        rref1, rref2 = rpc.remote("worker0", torch.add, args=(torch.ones(2,2),1)), rpc.remote("worker0", torch.add, args=(torch.zeros(2,2), 3))
        x = rref1.to_here() + rref2.to_here()
        print(x)
    rpc.join_rpc() # comment this and uncomment in init_process to remove error

def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29509'
    dist.init_process_group(backend, rank=rank, world_size=size, timeout=timedelta(seconds=12))
    rpc.init_model_parallel("worker{}".format(rank))
    fn(rank, size)
   #  rpc.join_rpc()

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()