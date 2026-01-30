import os
import torch
import torch.distributed as dist
import torch.utils.benchmark as benchmark

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
dist.init_process_group(backend="nccl")

x = torch.randn(1024, 1024, device='cuda')

if dist.get_rank() == 0:
    dist.send(x[0], 1)
elif dist.get_rank() == 1:
    dist.recv(x[0], 0)