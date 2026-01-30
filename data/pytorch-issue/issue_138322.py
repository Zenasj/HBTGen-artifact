import time
import os
import torch
from datetime import timedelta

rank = int(os.getenv("RANK", "0"))
torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=1))

if rank != 3:
    torch.distributed.barrier()

print(f"{rank} Done.")

