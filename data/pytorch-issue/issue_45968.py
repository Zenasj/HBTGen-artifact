import torch.nn as nn

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def main(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model = torch.nn.Linear(512, 4096).to(rank)

    ddp_model = DDP(model, device_ids=[rank])
    ddp_model._register_comm_hook(state=None, hook=fp16_compress_hook) 
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-4)

    for _ in range(5):
        y = ddp_model (torch.randn(64, 512, device=rank)).mean()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print(rank, torch.cuda.memory_allocated(device=rank) / (1024 ** 2))
    dist.destroy_process_group()

# as per https://github.com/pytorch/pytorch/blob/master/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py
def fp16_compress_hook(process_group: object, bucket: dist._GradBucket):
    """
        This DDP communication hook implements a simple gradient compression
        approach that converts ``GradBucket`` tensors whose type is assumed to be
        ``torch.float32`` to half-precision floating point format (``torch.float16``).
        It allreduces those ``float16`` gradient tensors. Once compressed gradient
        tensors are allreduced, its then callback called ``decompress`` converts the
        aggregated result back to ``float32`` and takes the mean.
        Example::
            >>> ddp_model._register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = (
        process_group.size() if process_group is not None else dist.get_world_size()
    )

    compressed_tensor = bucket.get_tensors()[0].to(torch.float16)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        return [fut.value()[0].to(torch.float32).div_(world_size)]

    return fut.then(decompress)

if __name__ == "__main__":
    mp.spawn(main, args=(2,), nprocs=2, join=True)