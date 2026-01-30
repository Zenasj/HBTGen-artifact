import os

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

def test_adamw(rank, world_size):
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    sharding_mesh=init_device_mesh("cuda", (world_size,), mesh_dim_names=["shard"])
    param = distribute_tensor(torch.rand(1024 * 1024 * 1024, device="cuda"), device_mesh=sharding_mesh, placements=[Shard(0)]).requires_grad_(True)
    param.grad = distribute_tensor(torch.rand(1024 * 1024 * 1024, device="cuda"), device_mesh=sharding_mesh, placements=[Shard(0)])
    optimizer = torch.optim.AdamW([param], lr=0.000001, fused=True)
    print(param.shape)

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True, record_shapes=True
        ) as prof:
        optimizer.step()

    if rank == 0:
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9876"

    world_size = 2

    torch.multiprocessing.spawn(
        test_adamw, args=(world_size,), nprocs=world_size, join=True
    )