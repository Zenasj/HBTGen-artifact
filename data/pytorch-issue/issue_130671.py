# Modified from https://github.com/pytorch/pytorch/tree/main/torch/distributed/_tensor
# to run this file (i.e. dtensor_example.py):
# torchrun --standalone --nnodes=1 --nproc-per-node=1 dtensor_example.py
import os
import torch
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor

mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))

big_tensor = torch.randint(256, size=(1024, 2048))
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
my_dtensor << 4