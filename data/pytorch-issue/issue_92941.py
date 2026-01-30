import torch.nn as nn

import torch
import os
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    dist.init_process_group("nccl")


class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super(MyModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        tmp = self.net(x)
        return torch.where(tmp <= 0.5, 0.4, 1.0)


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            *[nn.Linear(10, 100000)]
            + [MyModule(100000, 5)]
        )

    def forward(self, x):
        return self.net(x)

setup(0, 1)
model = ToyModel()
model.cuda()
inputs = (torch.randn(20, 10, device="cuda"),)

omodel = torch.compile(DDP(model))
omodel(*inputs)

import torch
from torch._dynamo.utils import deepcopy_to_fake_tensor

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_net_1_net_0 = torch.nn.Linear(in_features=100000, out_features=5, bias=True)
        self.self_net_1_net_1 = torch.nn.ReLU()

    def forward(self, self_net_0):
        self_net_1_net_0 = self.self_net_1_net_0(self_net_0);  self_net_0 = None
        self_net_1_net_1 = self.self_net_1_net_1(self_net_1_net_0);  self_net_1_net_0 = None
        le = self_net_1_net_1 <= 0.5;  self_net_1_net_1 = None
        where = torch.where(le, 0.4, 1.0);  le = None
        return where

m = Module()
input = torch.randn((20, 100000))

fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

fake_m = deepcopy_to_fake_tensor(m, fake_mode)
fake_input = fake_mode.from_tensor(input)
out = fake_m(fake_input)

Found in aten.where.self(
   *(
      FakeTensor(FakeTensor(..., device='meta', size=(20, 5), dtype=torch.bool), cpu),
      tensor(0.4000),
      tensor(1.)), 
    **{}
)