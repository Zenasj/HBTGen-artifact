import torch.nn as nn

import os
import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=2, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    with device:
        model = torch.nn.Linear(10, 10)
        input_ids = torch.randn(2, 10)
    model = FSDP(model)

    with torch.inference_mode():
        print("Validating ...")
        model(input_ids)
    print("Training ...")
    model(input_ids)

if __name__ == "__main__":
    mp.spawn(work, nprocs=2)