for _ in range(stage.get_devices_num()):
        gather_tensor.append(torch.zeros_like(in_slice))
dist.all_gather(gather_tensor, in_slice.contiguous(), group=group)

import os, sys
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    x = torch.ones(2, 2) * rank;
    outputs = []
    outputs.append(torch.zeros_like(x))
    outputs.append(torch.zeros_like(x))
    dist.all_gather(outputs, x)

    print("rank ", rank, ": ", outputs)
    sys.stdout.flush()

def init_processes(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

import os, sys
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size, x):
    # The result is error when dim is 0
    input = torch.chunk(x, size, dim=0)[rank]
    # The result is right when dim is not 0
    # input = torch.chunk(x, size, dim=1)[rank]
    # The result is right when use clone
    # input = torch.chunk(x, size, dim=0)[rank].clone()
    outputs = []
    outputs.append(torch.zeros_like(input))
    outputs.append(torch.zeros_like(input))
    dist.all_gather(outputs, input)

    print("rank ", rank, ": ", outputs)
    sys.stdout.flush()

def init_processes(rank, size, fn , x, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, x)


if __name__ == "__main__":
    size = 2
    processes = []
    x = torch.reshape(torch.tensor(range(4), dtype=torch.float32), [2, 2])
    print(x)
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run, x))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()