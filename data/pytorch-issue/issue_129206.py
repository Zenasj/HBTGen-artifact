import os

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
import torch.distributed.device_mesh
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh


torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))


mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        n = 8
        offset = mesh["tp"].get_local_rank() * n * n
        tensor = torch.arange(n * n, device=torch.cuda.current_device(), dtype=torch.bfloat16).view(n, n) + offset

        self.register_parameter(
            "col", nn.Parameter(DTensor.from_local(tensor, device_mesh=mesh["tp"], placements=[Shard(0)]))
        )

        self.register_parameter(
            "row", nn.Parameter(DTensor.from_local(tensor, device_mesh=mesh["tp"], placements=[Shard(1)]))
        )


model = DummyModel()
model = fully_shard(model, mesh=mesh["dp"])

# dcp.save(get_model_state_dict(model), checkpoint_id="tmp") # error also occurs with unsharding after dcp

for param in model.parameters():
    param = param.full_tensor()
    if torch.distributed.get_rank() == 0:
        print(param)