import torch.nn as nn

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import torch
from torchao.float8 import (
    CastConfig,
    Float8LinearConfig,
    ScalingType,
    convert_to_float8_training,
)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.l(x)
    

mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("ddp", "fsdp", "abcd"))

with torch.device(torch.cuda.current_device()):
    model = M()
    x = torch.randn(2, 2, device=torch.cuda.current_device())

model = convert_to_float8_training(
    model,
    config=Float8LinearConfig(
        enable_fsdp_float8_all_gather=True,
        cast_config_input=CastConfig(scaling_type=ScalingType("dynamic")),
        cast_config_weight=CastConfig(scaling_type=ScalingType("dynamic")),
        cast_config_grad_output=CastConfig(scaling_type=ScalingType("dynamic")),
        force_recompute_fp8_weight_in_bwd=True,
    ),
    module_filter_fn=lambda mod, fqn: fqn != "output",
)
model = fully_shard(model, mesh=mesh["ddp", "fsdp"])

y = model(x)