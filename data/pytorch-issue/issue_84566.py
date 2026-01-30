import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

def work(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print("finished creating process group")
    t_13 = torch.zeros(13)
    t_6 = torch.zeros(6)
    output_tensor = torch.zeros(13)
    if rank == 0:
        dist.scatter(output_tensor, [t_13, t_6])
    else:
        dist.scatter(output_tensor)
    print(f"rank {rank} - {output_tensor=}")

if __name__ == "__main__":
    world_size = 2
    mp.spawn(work, nprocs=world_size, args=(world_size,))