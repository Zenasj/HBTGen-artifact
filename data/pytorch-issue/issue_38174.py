import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import torch.nn as nn
def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    model = nn.Linear(1, 1, bias=False).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    # Create uneven inputs, rank 1 will get one more input than rank 0. This will cause a hang.
    inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
    for _ in range(5):
        for inp in inputs:
            loss = model(inp).sum()
            loss.backward()
    torch.cuda.synchronize(device=rank)

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost" ; os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=2, args=())