import torch
from memory_profiler import profile

@profile
def test():
    print("before")
    torch.distributed.barrier()
    print("after")

torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1, init_method='tcp://localhost:23456')

test()