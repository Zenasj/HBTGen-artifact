import torch.nn.functional as F

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed.tensor import Shard, DTensor, Replicate
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
_world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
conv = nn.Conv2d(64, 64, 3, padding=1).train()
x = torch.randn(1, 64, 32, 32)
x_dt = DTensor.from_local(x, device_mesh, [Replicate()])
w = conv.weight.data
w_dt = torch.nn.Parameter(DTensor.from_local(w, device_mesh, [Replicate()]))

b = conv.bias.data
b_dt = torch.nn.Parameter(DTensor.from_local(b, device_mesh, [Replicate()]))

res = F.conv2d(x_dt, w_dt, b_dt, padding=1)
res_l = res.to_local()
dres = torch.rand_like(res_l)
res_l.backward(dres)

dist.barrier()
dist.destroy_process_group()