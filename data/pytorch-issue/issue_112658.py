import torch.nn as nn

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch import distributed
import os

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
distributed.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
device = torch.device('cuda', local_rank)

print(local_rank, world_size, device)

model = nn.Linear(100,100)
model.to(device)
model = FSDP(model, device_id=local_rank)
#torch.cuda.set_device(device)
output = model(torch.randn((1, 100), device=device))