import os
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
import torch.multiprocessing as mp


class UnitModule(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100, device=device),
            nn.ReLU(),
        )
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        return self.l2(self.seq(self.l1(x)))

class CompositeParamModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.p = nn.Parameter(torch.randn((100, 100), device=device))
        self.register_buffer(
            "buffer", torch.randn((100, 100), device=device), persistent=True
        )

    def forward(self, x):
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)

def localTrain():
    model = CompositeParamModel(device=torch.device("cuda"))
    model_state_dict1 = get_model_state_dict(model)
    key = next(iter(model_state_dict1.keys()))
    model_state_dict1.pop(key)
    model_state_dict1["abc"] = torch.zeros(10)
    set_model_state_dict(model, model_state_dict=model_state_dict1)

def worker(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=2)
    model = CompositeParamModel(device=torch.device("cuda"))
    model = FSDP(model)
    model_state_dict1 = get_model_state_dict(model)
    key = next(iter(model_state_dict1.keys()))
    # model_state_dict1.pop(key)
    model_state_dict1.clear()
    model_state_dict1["abc"] = torch.zeros(10)
    # set_model_state_dict(model, model_state_dict=model_state_dict1)
    model.load_state_dict(model_state_dict1, strict=True)

def distTrain():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=2, args=())

if __name__ == '__main__':
    distTrain()