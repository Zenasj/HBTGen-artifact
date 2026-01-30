import io
import os
import pprint
import sys
import torch.distributed as dist
from datetime import timedelta
import torch.distributed as dist

if __name__ == "__main__":
    os.environ["MASTER_PORT"]="29501"
    dist.init_process_group(backend="gloo")
    dist.barrier()

import io
import os
import pprint
import sys
from datetime import timedelta
import torch.distributed as dist


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "<eth0 interface ip of node 1>"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo")
    dist.barrier()

import os
import torch.distributed as dist
import datetime
import sys
dist.init_process_group(backend="nccl")
print(f"[ {os.getpid()} ] world_size = {dist.get_world_size()}, " + f"rank = {dist.get_rank()}, backend={dist.get_backend()}")