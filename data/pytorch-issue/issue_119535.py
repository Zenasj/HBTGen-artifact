import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
device = torch.device('cuda', dist.get_rank())

model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.BatchNorm1d(10), nn.Linear(10, 10))
model.to(device)
model = DDP(model)

options = StateDictOptions(cpu_offload=True, strict=True)
model_state_dict = get_model_state_dict(model, options=options)
set_model_state_dict(model, model_state_dict, options=options)