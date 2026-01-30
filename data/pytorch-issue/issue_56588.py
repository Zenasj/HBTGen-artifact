import os
import torch
from torch import multiprocessing as mp
from torch.distributed import rpc

def foo(with_rpc=True):
    device = torch.cuda.current_device()
    prefix = 'with_rpc' if with_rpc else 'without_rpc'
    print(f'{prefix}: {os.getpid()=}, {device=}')

def worker0():
    rpc.init_rpc("worker0", rank=0, world_size=2)
    foo(with_rpc=False)
    rpc.remote("worker1", foo)
    rpc.shutdown()

def worker1():
    rpc.init_rpc("worker1", rank=1, world_size=2)
    foo(with_rpc=False)
    rpc.remote("worker0", foo)
    rpc.shutdown()

def run(rank):
    device = rank
    torch.cuda.set_device(rank)
    assert torch.cuda.current_device() == device
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if rank == 0:
        worker0()
    else:
        worker1()


def main():
    mp.start_processes(run, nprocs=2, start_method='fork')


if __name__ == '__main__':
    main()