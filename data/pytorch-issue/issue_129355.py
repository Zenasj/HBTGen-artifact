import random

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput

device = "cuda"
_world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type=device, mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.l1 = nn.Linear(n_embd, 12)
        self.l2 = nn.Linear(12, 11)

    def forward(self, x):
        x0 = self.ln1(x) # inside LayerNorm.forward the DTensor shape is (B, T*_world_size, n_embed), I'd expect (B, T/_world_size, n_embed) instead!
        x1 = self.l1(x0)
        x2 = F.relu(x1)
        x3 = self.l2(x2)
        return x3

B,T,n_embd = 64, 16, 1024

data = torch.randn(B,T,n_embd).to(device)
model = MyModel().to(device)

model = parallelize_module(
    model,
    device_mesh=device_mesh,
    parallelize_plan={
        "ln1": SequenceParallel(use_local_output=False),
        "l1": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),), 
            ),
        "l1": ColwiseParallel(),
        "l2": RowwiseParallel(),
    }
)

out = model(data)
destroy_process_group()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput

device = "cuda"
_world_size = int(os.environ.get("WORLD_SIZE",-1))
if _world_size > -1:
    device_mesh = init_device_mesh(device_type=device, mesh_shape=(_world_size,))
    _rank = device_mesh.get_rank()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.l1 = nn.Linear(n_embd, 12)
        self.l2 = nn.Linear(12, 11)

    def forward(self, x):
        x0 = self.embed(x)
        x1 = self.ln1(x0)
        x2 = self.l1(x1)
        x3 = F.relu(x2)
        x4 = self.l2(x3)
        return x4

B,T,n_embd = 64, 2048, 4096

model = MyModel().to(device)

if _world_size > -1:
    model = parallelize_module(
        model,
        device_mesh=device_mesh,
        parallelize_plan={
            "embed": ColwiseParallel(output_layouts=Shard(1), use_local_output=False),
            "ln1": SequenceParallel(use_local_output=False),
            "l1": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ), 
            "l1": ColwiseParallel(),
            "l2": RowwiseParallel(output_layouts=Shard(1)),
        }
    )

data = torch.randint(0, 100, (B,T)).to(device)
out = model(data)

if _world_size > -1: destroy_process_group()

import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput


random.seed(0)
torch.manual_seed(0)

device = "cuda"
_world_size = int(os.environ.get("WORLD_SIZE",-1))
if _world_size > -1:
    device_mesh = init_device_mesh(device_type=device, mesh_shape=(_world_size,))
    _rank = device_mesh.get_rank()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.l1 = nn.Linear(n_embd, 12)
        self.l2 = nn.Linear(12, 11)

    def forward(self, x):
        x0 = self.embed(x)
        x1 = self.ln1(x0)
        x2 = self.l1(x1)
        x3 = F.relu(x2)
        x4 = self.l2(x3)
        return x4

B,T,n_embd = 64, 2048, 4096

model = MyModel().to(device)

if _world_size > -1:
    model = parallelize_module(
        model,
        device_mesh=device_mesh,
        parallelize_plan={
            "embed": ColwiseParallel(output_layouts=Shard(1), use_local_output=False),
            "ln1": SequenceParallel(use_local_output=False),      
            "l1": ColwiseParallel(input_layouts=Shard(1)),
            "l2": RowwiseParallel(output_layouts=Replicate()),
        }
    )

import time
t0 = time.time()

for _ in range(3):
    data = torch.randint(0, 100, (B,T)).to(device)
    out = model(data)
    loss = out.sum()
    loss.backward()

torch.cuda.synchronize()
t1 = time.time()
print(f'Elapsed time: {1000*(t1-t0):.2f}ms')

if _world_size > -1: destroy_process_group()