py
import os

import torch
import torch.distributed
import torch.multiprocessing as mp


def run(rank):
    torch.distributed.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:50001", rank=rank, world_size=4
    )
    torch.cuda.set_device(rank)
    tensor = torch.tensor([rank], device="cuda")
    torch.distributed.broadcast(tensor, src=0)
    
    raise Exception("Exception happened somewhere in user code!")
    

if __name__ == "__main__":
    mp.spawn(run, nprocs=4)