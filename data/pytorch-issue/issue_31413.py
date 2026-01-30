def run_sparse(rank, size):
    a = torch.randn(5) * (rank + 1)
    mask = a > 0  # sparse mask
    i = mask.nonzero()  # value indexes
    v = a[mask]  # sparse values
    t = torch.sparse.FloatTensor(i.t(), v, a.size())

    pprint("Before\t", t.to_dense())
    dist.all_reduce(t)
    pprint("After\t", t.to_dense())

import os, time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def pprint(*msg):
    print("Message from %s" % rank, *msg)

def run(rank, size):
    t = torch.ones(5) * (rank + 1)
    pprint("Before\t", t)
    dist.all_reduce(t)
    pprint("After\t", t)

def run_sparse(rank, size):
    a = torch.randn(5) * (rank + 1)
    mask = a > 0  # sparse mask
    i = mask.nonzero()  # value indexes
    v = a[mask]  # sparse values
    t = torch.sparse.FloatTensor(i.t(), v, a.size())

    pprint("Before\t", t.to_dense())
    dist.all_reduce(t)
    pprint("After\t", t.to_dense())


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_sparse))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()