import torch.nn as nn

# file: torch_compile_error.py
import torch

def add_tensors(a, b):
    c = a + b
    return c

a = torch.tensor([1, 2, 3]).to("cuda")
b = torch.tensor([1, 2, 3]).to("cuda")
c = add_tensors(a, b)

print(f"{a=}, {b=}, {c=}")

# Compile
add_tensors_compiled = torch.compile(add_tensors)

c_compiled = add_tensors_compiled(a, b)
print("{c_compiled=}")

from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config









from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
        l_a_ = L_a_
        l_b_ = L_b_
        c = l_a_ + l_b_;  l_a_ = l_b_ = None
        return (c,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('e2d1839ed1706f7d470d87f8c48a5584cafa5a12', 24, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (3,), dtype=torch.int64, is_leaf=True)  # L_a_
    buf1 = reader.storage('e2d1839ed1706f7d470d87f8c48a5584cafa5a12', 24, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (3,), dtype=torch.int64, is_leaf=True)  # L_b_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify',
        save_dir='/mnt/pv/notebooks/smishra/torch_compile_debug/run_2024_02_02_10_49_36_213095-pid_366606/minifier/checkpoints', autocast=False, backend='inductor')