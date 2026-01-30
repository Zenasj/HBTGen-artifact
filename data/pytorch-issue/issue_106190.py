import torch.distributed as dist
import torch
import os
import time
import json
import sys

def bm_all_gather(shape, count=None):
    world_size = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK"))
    data = torch.randn(shape, dtype=torch.float32).to(f'cuda:{local_rank}')
    tensor_list = [torch.zeros_like(data).to(f'cuda:{local_rank}') for _ in range(world_size)]
    dst = torch.zeros_like(data).to(f'cuda:{local_rank}')
    for _ in range(10):
        dist.all_gather(tensor_list, data)
    s = time.time()
    start = int(s)
    t = []
    stop_time = int(os.getenv('stop_time', 600))
    #for i in range(100000):
    while True:
        start = time.time()
        for j in range(10):
            #dist.all_reduce(tensor_list, data)
            dist.all_reduce(data)
        end = time.time()


def main():
    bm_all_gather(1<<20)

if __name__ == "__main__":
    local_rank = int(os.getenv("LOCAL_RANK"))
    global_rank = int(os.getenv("GLOBAL_RANK"))
    torch.distributed.init_process_group(backend="nccl", world_size=64, rank=global_rank)
    torch.cuda.set_device(local_rank)
    main()