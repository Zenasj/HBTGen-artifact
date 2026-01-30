import copy
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

dist.init_process_group(backend="nccl", init_method="env://")
model = nn.Linear(3, 1024)
fsdp_model = FSDP(model, device_id=0, auto_wrap_policy=size_based_auto_wrap_policy)
fsdp_model_ema = copy.deepcopy(fsdp_model)