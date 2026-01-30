import torch
import os
import torch.distributed as dist

def repro(rank, world_size):
    device=torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    device_mesh = dist.init_device_mesh(
        "cuda", (2, world_size // 2)
    )
    dist.destroy_process_group()
    print("clean exit")

if __name__ == "__main__":
    repro(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))