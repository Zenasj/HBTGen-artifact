# torchrun --standalone --nproc_per_node=4 run_hsdp2.py

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import torch
import copy
torch.cuda.memory._record_memory_history(max_entries=100000)


def main():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=('dp_replicate', 'dp_shard'))

    allocate_cublas_workspace()

    dim0 = 4096
    torch.manual_seed(0)
    vanilla_model = nn.Sequential(
        *[torch.nn.Linear(dim0, dim0, bias=False, device="cpu") for _ in range(1)]
    )
    dist_model = copy.deepcopy(vanilla_model)
    for layer in dist_model:
        fully_shard(layer, mesh=mesh)
    fully_shard(dist_model)
    dist_optim = torch.optim.AdamW(dist_model.parameters(), lr=0.05)
    
    x = torch.randn((8192, dim0), device="cuda")
    for _ in range(6):
        dist_optim.zero_grad()
        dist_loss = dist_model(x).sum()
        dist_loss.backward()
    
    if torch.distributed.get_rank() == 0:
        torch.cuda.memory._dump_snapshot("test_hsdp2.pickle")


def allocate_cublas_workspace():
    a = torch.randn(1, 1, device="cuda", requires_grad=True)
    b = torch.randn(1, 1, device="cuda", requires_grad=True)
    return torch.matmul(a, b).sum().backward()


if __name__ == "__main__":
    main()