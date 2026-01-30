# python -m torch.distributed.run --nproc_per_node=2 <script>.py

import torch
import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group("nccl")
    dist.broadcast(torch.tensor([1, 2, 3]).cuda(), 0)