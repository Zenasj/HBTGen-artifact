import torch.nn as nn

ComputedBuffer(name='buf30', layout=FixedLayout('cpu', torch.float16, size=[3, 4], stride=[4, 1]), data=Pointwise(
  'cpu',
  torch.float16,
  to_dtype(load(arg11_1, 4*i0 + i1, False), torch.float16),
  ranges=[3, 4],
  origins={_to_copy_default}
))

import torch
from torch import tensor, device
import torch.fx as fx
from torchdynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.13.0a0+gite09821f
# torch cuda version: 11.6
# torch git version: e09821f784bc9e9f13d361e9d2eb3fa1d7d07263


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Tue_Mar__8_18:18:20_PST_2022 
# Cuda compilation tools, release 11.6, V11.6.124 
# Build cuda_11.6.r11.6/compiler.31057947_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3090 : 1 

class Repro(torch.nn.Module):



    def forward(self, arg0_1, arg1_1):
        index_tensor = torch.ops.aten.index.Tensor(arg1_1, [arg0_1]);  arg1_1 = arg0_1 = None
        return (index_tensor,)
    
args = [((0,), (1,), torch.int64, 'cuda'), ((5000, 4), (4, 1), torch.float16, 'cuda')]
args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
mod = make_fx(Repro())(*args)

from torchinductor.compile_fx import compile_fx_inner

compiled = compile_fx_inner(mod, args)
compiled(*args)