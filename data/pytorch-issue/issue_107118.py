import torch.nn as nn

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1, convert_element_type):
        cat = torch.ops.aten.cat.default([arg0_1, convert_element_type], 1);  arg0_1 = convert_element_type = None
        return (cat,)
        
args = [((1, 4, 32, 32), (4096, 1, 128, 4), torch.float32, 'cuda'), ((0,), (1,), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced