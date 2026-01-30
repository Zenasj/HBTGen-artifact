import torch.nn as nn

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, get_optimizer_state_dict
mod = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
opt = torch.optim.SGD(mod.parameters(), momentum=0.9)

mod(torch.randn(1, 3)).sum().backward()
opt.step()

sd = get_optimizer_state_dict(mod, opt)
print(sd)

sd['state'].pop('0.weight')

set_optimizer_state_dict(mod, opt, sd)  # fail
# opt.load_state_dict(sd)  # succeed
sd = get_optimizer_state_dict(mod, opt)
print(sd)

from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
dist.init_process_group(world_size=1, rank=0, backend='gloo')

mod = DDP(nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3)))

model_state_dict = mod.state_dict()
model_state_dict.pop('module.0.weight')
options = StateDictOptions(
    strict=False,
)
set_model_state_dict(mod, model_state_dict, options=options) # fail