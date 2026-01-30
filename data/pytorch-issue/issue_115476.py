import torch
a = torch.Tensor([2,3])
print(a)

import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor,zeros

def run(rank, size):

  a = torch.tensor([[0, 2.], [3, 0]])
  a.neg()


def init_process(rank_id, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12347'
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL" 
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size)

if __name__ == "__main__":
    big_tensor = torch.arange(0,16).reshape(4,4)
    size = 1
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()