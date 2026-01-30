import torch.nn as nn

import os
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_matrix(r):
    if dist.get_rank() == 1:
        print("r.device inside call:", r.device, "rank:", dist.get_rank())

    x, y, z = r.unbind(-1)
    R = torch.stack([-z,  y,
                      z, -x,
                     -y,  x], dim=-1)
    return R.reshape(*x.shape, 3, 2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        num_views = 150000
        self.pose_update_r = nn.Parameter(torch.zeros(num_views, 3))
        nn.init.zeros_(self.pose_update_r)

    def forward(self, idx):
        # Triggers error 0
        # r = self.pose_update_r.new_zeros((idx.shape[0], 3))

        # Triggers error 1
        r = F.embedding(idx, self.pose_update_r) # (B, 3)

        # Triggers error 2
        # r = F.embedding(idx, self.pose_update_r, max_norm=0.175) # (B, 3)

        if dist.get_rank() == 1:
            print("r.device before call:", r.device, "rank:", dist.get_rank())
        T = get_matrix(r)
        return T


def train(rank, world_size):
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = Model().to(rank)
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_ddp = torch.compile(model_ddp)

    for _ in tqdm(range(1000)):
        idx = torch.randint(1000, (8,)).to(rank)
        T = model_ddp(idx)

    cleanup()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)