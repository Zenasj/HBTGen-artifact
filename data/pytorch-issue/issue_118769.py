from typing import Tuple, List, Dict, cast
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import distribute_tensor, DTensor, Shard, Placement, Replicate

mesh = init_device_mesh(device_type="cuda", mesh_shape=(2,))
x = torch.randn(4, 8, requires_grad=True)
y = torch.randn(4, 32, requires_grad=True)
x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)
from torch.distributed._tensor.debug import CommDebugMode
comm_mode = CommDebugMode()
with comm_mode:
    z = torch.mm(x_dtensor, y_dtensor)
print(comm_mode.get_comm_counts())