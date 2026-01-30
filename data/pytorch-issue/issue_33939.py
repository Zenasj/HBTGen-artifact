import multiprocessing as mp
import torch
from torch.distributed import rpc
import os

def my_mm(a, b):
    return torch.mm(a, b)

def my_spmm(a, b):
    return torch.spmm(a, b)

def _run_process(rank, world_size):
    name = "worker{}".format(rank)
    # Initialize RPC.
    rpc.init_rpc(name=name,rank=rank,world_size=world_size)
    if rank == 0:
        i = torch.LongTensor([[0, 1, 1],[2, 0, 2]])
        v = torch.FloatTensor([3, 4, 5])
        t1 = torch.sparse.FloatTensor(i, v, torch.Size([2,3]))
        t2 = torch.Tensor([[1,4],[7,9],[6,3]])
        result1 = rpc.rpc_async("worker1", my_mm, args=(t1.to_dense(), t2))
        print (result1.wait())
        result2 = rpc.rpc_async("worker1", my_spmm, args=(t1, t2))
        print (result2.wait())

def run_process(rank, world_size):
    _run_process(rank, world_size)
    rpc.shutdown()

processes = []
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
# Run world_size workers.
world_size = 2
for i in range(world_size):
    p = mp.Process(target=run_process, args=(i, world_size))
    p.start()
    processes.append(p)

for p in processes:
    p.join()