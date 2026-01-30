import os

import torch.distributed as dist

world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=world_size,
    rank=rank,
)
dist.destroy_process_group()

print("----------destroy finish--------------")
# I try to change the connect port as follows. It also does not help.
# os.environ["MASTER_PORT"] = "22222"
dist.init_process_group(    ## get stuck here
    backend="nccl",
    init_method="env://",
    world_size=world_size,
    rank=rank,
)
dist.destroy_process_group()

import gc
gc.collect()