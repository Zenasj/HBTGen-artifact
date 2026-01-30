# torchrun --nproc_per_node=2 crash.py
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, DTensor

mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('ep',))
dt = DTensor.from_local(torch.randn(2, 4, device='cuda'), mesh, [Shard(0)]).requires_grad_()
def f(dt): return dt.redistribute(placements=[Shard(1)]).to_local()
f(dt).sum().backward() # no crash
f = torch.compile(f)
f(dt).sum().backward() # crash

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _register_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._collective_utils import _shard_dim_alltoall_meta, _get_group_size_by_name

mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('ep',))
_register_process_group('ep', mesh['ep'].get_group())
x = torch.randn(2, 4, device='meta')
y = _shard_dim_alltoall_meta(x, 0, 1, 'ep')
if dist.get_rank() == 0:
    print(x.shape, x.stride()) # torch.Size([2, 4]) (4, 1)
    print(y.shape, y.stride()) # torch.Size([4, 2]) (4, 1)