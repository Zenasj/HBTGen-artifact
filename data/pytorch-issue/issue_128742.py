import torch.nn as nn

"""
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 test_setattr.py >artifacts/test_output2.txt 2>&1
"""

import functools
import torch
from torch.distributed._tensor import DTensor, Replicate, init_device_mesh

class FSDPParam:
    def __init__(self):
        device_mesh = init_device_mesh("cuda", (1,))
        replica_placement = [Replicate()]
        local_tensor = torch.zeros(3, 3, device="cuda")
        dtensor = DTensor.from_local(local_tensor, device_mesh=device_mesh, placements=replica_placement)
        self.sharded_param = torch.nn.Parameter(dtensor)

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = torch.nn.Parameter(torch.ones(3, 3, device="cuda"))

    def forward(self, x):
        return x + self.foo

def forward_post_hook(module, input, output, _fsdp_param):
    setattr(module, "foo", _fsdp_param.sharded_param)

# Eager test
# fsdp_param = FSDPParam()
# mod = TestModule()
# mod.register_forward_hook(functools.partial(forward_post_hook, _fsdp_param=fsdp_param))
# inp = torch.zeros(3, 3)
# mod(inp)
# assert torch.allclose(mod.foo.sum(), torch.tensor(0.))

# Compile test
fsdp_param = FSDPParam()
mod = TestModule()
mod.register_forward_hook(functools.partial(forward_post_hook, _fsdp_param=fsdp_param))
compiled_mod = torch.compile(mod, fullgraph=True)
inp = torch.zeros(3, 3, device="cuda")
compiled_mod(inp)
assert torch.allclose(compiled_mod.foo.sum(), torch.tensor(0.))