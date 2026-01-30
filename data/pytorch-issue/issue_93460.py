import torch.nn as nn

import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# REPLACEABLE COMMENT FOR TESTING PURPOSES

# torch version: 1.14.0a0+git76ba93c
# torch cuda version: None
# torch git version: 76ba93c1cb4e9584e749e0a51bdfbe7bf186df90


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, cat):
        unsqueeze_ = torch.ops.aten.unsqueeze_.default(arg0_1, 0);  arg0_1 = None
        return (cat,)
        
args = [((1, 1, 1, 12, 11, 3), (396, 396, 396, 33, 3, 1), torch.int64, 'cpu'), ((1, 1, 1, 12, 11, 3), (396, 396, 396, 33, 3, 1), torch.int64, 'cpu')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro())(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)