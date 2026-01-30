import torch.nn as nn

import torch 
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
)
from torch.distributed.tensor import Shard, DTensor, Replicate
import os
_world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
class TestMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 64)
    
    def forward(self, x):
        return self.fc(x)
mod = TestMod().cuda()
parallelize_module(mod, device_mesh, {
    "fc": ColwiseParallel(use_local_output=False)
})
optim = torch.optim.AdamW([
    {"params": mod.parameters()},
    {"params": [], "lr": 0.2}, # empty pg group here
], lr=0.1)
optim_new = torch.optim.AdamW([
    {"params": mod.parameters()},
    {"params": [], "lr": 0.2}, # empty pg group here
], lr=0.1)

# init optimizer state
sample_inp = torch.randn(2, 128, 64).cuda()
sample_target = torch.randn(2, 128, 64).cuda()
loss_cls = torch.nn.MSELoss()
optim.zero_grad()
output = mod(sample_inp).redistribute(device_mesh, [Replicate()]).to_local()
loss = loss_cls(output, sample_target)
loss.backward()
optim.step()
# bug
optim_state_dict = get_optimizer_state_dict(mod, optim)
set_optimizer_state_dict(mod, optim_new, optim_state_dict)

optim_new.step()

torch.distributed.checkpoint.state_dict.set_optimizer_state_dict(
                model=self._module,
                optimizers=self._optimizer,
                optim_state_dict=optimizer_state_dict,
                options=torch.distributed.checkpoint.state_dict.StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                    ignore_frozen_params=False,
                    broadcast_from_rank0=True,
                ),
            )