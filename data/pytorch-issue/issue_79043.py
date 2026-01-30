import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def main(rank, world_size):

    #use the default pytorch port
    port = "29500"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend = "nccl", rank = rank, world_size = world_size)

    # set device
    comm_rank = dist.get_rank()
    comm_local_rank = comm_rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{comm_local_rank}")

    # create tensor
    tensor = torch.ones((2, 16, 768, 1152), dtype=torch.float32, device=device).to(memory_format=torch.channels_last)

    print("Is contiguous:", tensor.is_contiguous(memory_format=torch.channels_last))

    print("Broadcasting now")

    # broadcast
    dist.broadcast(tensor, src=0)

    print("All Done!")

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)                                                                                                                                                       

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    run_demo(main, world_size)