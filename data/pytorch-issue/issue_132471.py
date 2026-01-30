import torch.nn as nn

from torch.distributed._composable.fsdp import fully_shard
import torch.distributed as dist
from torch import nn

if __name__ == '__main__':
    dist.init_process_group(backend='gloo')
    model = nn.Sequential(nn.Linear(100, 100))
    print(model[0].weight.device)
    model_fsdp = fully_shard(model)
    dtensor = model_fsdp[0].weight
    print(dtensor.device)
    print(dtensor.full_tensor())