import os
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint import save

def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=2, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    t = torch.tensor([0, 1] if rank == 0 else [2, 3], device=device)
    state = {"weight": t}
    save(state, checkpoint_id="repro_ckpt")

    if rank == 0:
        print("__0_0.distcp", torch.load("repro_ckpt/__0_0.distcp"))  # [0, 1]
        print("__1_0.distcp", torch.load("repro_ckpt/__1_0.distcp"))  # [2, 3]

def run():
    mp.spawn(work, nprocs=2)

if __name__ == "__main__":
    run()

state = {f"{rank}.weight": t}

from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import init_device_mesh

device_mesh = init_device_mesh("cuda", (2,))
placements = [Shard(0) for _ in range(device_mesh.ndim)]
t = DTensor.from_local(t, device_mesh, placements)