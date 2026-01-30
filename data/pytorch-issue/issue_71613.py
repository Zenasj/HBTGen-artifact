import random

import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn as nn
import torch.nn.functional as F

class ComplexLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out
    ):
        super().__init__()
        linear = nn.Linear(dim, dim_out, dtype = torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(linear.weight))
        self.bias = nn.Parameter(torch.view_as_real(linear.bias))

    def forward(self, x):
        weight = torch.view_as_complex(self.weight)
        bias = torch.view_as_complex(self.bias)
        return F.linear(x, weight, bias)

class ModReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))

def example(rank, world_size):
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # create local model
    model = nn.Sequential(
        ComplexLinear(10, 10),
        ModReLU()
    ).to(rank)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    x = torch.view_as_complex(torch.rand(20, 10, 2)).to(rank)

    outputs = ddp_model(x)
    labels = torch.randn(20, 10).to(rank)

    # backward pass
    loss_fn(torch.abs(outputs), labels).backward()

    # update parameters
    optimizer.step()
    print(outputs)

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    os.environ['MASTER_ADDR'] =  'localhost' 
    os.environ['MASTER_PORT'] = str(np.random.randint(10000, 20000))
    main()

class ComplexConv2d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        stride = 1,
        padding = 0
    ):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype = torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))
        return F.conv2d(x, weight, bias, stride = self.stride, padding = self.padding)