import torch.nn as nn

var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True);  add = None

import torch._inductor.overrides
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx
# torch version: 1.13.0a0+git3eb2722
# torch cuda version: 11.4
# torch git version: 3eb27229dd74dd0bea434326c471f16c50e558a4
# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2021 NVIDIA Corporation 
# Built on Sun_Aug_15_21:14:11_PDT_2021 
# Cuda compilation tools, release 11.4, V11.4.120 
# Build cuda_11.4.r11.4/compiler.30300941_0 
# GPU Hardware Info: 
# NVIDIA A100-PG509-200 : 8 
from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, add):
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True);  add = None
        return (var_mean,)
        
args = [((1, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)
from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models
compiled = compile_fx_inner(mod, args)
assert same_two_models(mod, compiled, args, only_fwd=True), "Accuracy failed"