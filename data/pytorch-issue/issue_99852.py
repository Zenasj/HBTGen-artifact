import torch.nn as nn

import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.0a0+gita27bd42
# torch cuda version: 11.4
# torch git version: a27bd42bb9ad39504fdd94ad38a5ad0346f1758b


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2021 NVIDIA Corporation 
# Built on Sun_Aug_15_21:14:11_PDT_2021 
# Cuda compilation tools, release 11.4, V11.4.120 
# Build cuda_11.4.r11.4/compiler.30300941_0 

# GPU Hardware Info: 
# NVIDIA PG509-210 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, unsqueeze_1, unsqueeze_3, rsqrt, convolution):
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem_1 = var_mean[1];  var_mean = None
        sub = torch.ops.aten.sub.Tensor(convolution, getitem_1);  convolution = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_6 = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        gt = torch.ops.aten.gt.Scalar(add_4, 0);  add_4 = None
        return (gt,)
        
import torch._dynamo.repro.after_aot
reader = torch._dynamo.repro.after_aot.InputReader(save_dir='/tmp/minifier-20230423')
buf0 = reader.storage('b3e108077e73f8bbdefbd419a1798700731646a1', 256, device=device(type='cuda', index=0))
t0 = reader.tensor(buf0, (64, 1, 1))
buf1 = reader.storage('6ee108072a73f8bb41fbd4197ff98700151646a1', 256, device=device(type='cuda', index=0))
t1 = reader.tensor(buf1, (64, 1, 1))
buf2 = reader.storage('65d17de8e97efedb72fa1a01d7f26cc32f948b59', 256, device=device(type='cuda', index=0))
t2 = reader.tensor(buf2, (1, 64, 1, 1))
buf3 = reader.storage('38c695755b4a84a70b07c240f092e2b293280811', 50331648, device=device(type='cuda', index=0))
t3 = reader.tensor(buf3, (4, 64, 192, 256))
args = [t0, t1, t2, t3]
mod = make_fx(Repro(), tracing_mode='symbolic')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
class AccuracyError(Exception):
    pass
if not same_two_models(mod, compiled, args, only_fwd=True):
    raise AccuracyError("Bad accuracy detected")