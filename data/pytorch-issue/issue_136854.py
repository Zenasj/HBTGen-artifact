import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

MODEL_PARALLEL_SIZE = 2
CHECKPOINT_DIR = f"checkpoint"

def _init_model(process_group):
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU()).cuda()
    model = FSDP(
        model,
        process_group=process_group,
        use_orig_params=True,
    )

    return model

def run(rank, world_size, device="cuda"):
    dist.init_process_group(rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device_mesh = init_device_mesh(
        device,
        (MODEL_PARALLEL_SIZE, world_size // MODEL_PARALLEL_SIZE),
        mesh_dim_names=("mp", "dp")
    )

    model = _init_model(device_mesh.get_group("dp"))

    dcp.save(
        state_dict={"model": model},
        checkpoint_id=CHECKPOINT_DIR + f"/{device_mesh['mp'].get_local_rank()}",
        process_group=device_mesh.get_group("dp")
    )

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = torch.cuda.device_count()
    print(f"Running stateful checkpoint example on {world_size} devices.")
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )