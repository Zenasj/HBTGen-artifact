import torch.nn as nn

import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx
import sys

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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, add, getitem_1, reciprocal):
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, reciprocal);  sub = reciprocal = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view = torch.ops.aten.view.default(convert_element_type, [128, 1024]);  convert_element_type = None
        addmm = torch.ops.aten.addmm.default(arg3_1, view, permute);  arg3_1 = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [1, 128, 1024]);  addmm = None
        mul_2 = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg5_1, view, permute_1);  arg5_1 = view = permute_1 = None
        view_2 = torch.ops.aten.view.default(addmm_1, [1, 128, 1024]);  addmm_1 = None
        view_3 = torch.ops.aten.view.default(view_2, [1, -1, 16, 64]);  view_2 = None
        permute_2 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        clone = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_6 = torch.ops.aten.view.default(mul_2, [1, 128, 16, 64]);  mul_2 = None
        permute_5 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_2 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_7 = torch.ops.aten.view.default(clone_2, [16, -1, 64]);  clone_2 = None
        view_8 = torch.ops.aten.view.default(clone, [16, -1, 64]);  clone = None
        permute_6 = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        bmm = torch.ops.aten.bmm.default(view_7, permute_6);  view_7 = permute_6 = None
        return (bmm,)
        
args = [((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024, 1024), (1024, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024, 1024), (1024, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((1, 128, 1), (128, 1, 1), torch.float32, 'cuda'), ((1, 128, 1), (128, 1, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
try:    mod = make_fx(Repro().to(device="cuda"))(*args)
except:    sys.exit(0)
from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

try:
    compiled = compile_fx_inner(mod, args)
except:
    sys.exit(0)
assert same_two_models(mod, compiled, args, only_fwd=True), "Accuracy failed"

import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.14.0a0+fb
# torch cuda version: 11.4.0
# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-PG509-200 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg12_1, arg13_1, arg158_1, arg159_1, add_28, mul_20, mm_36):
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg12_1);  mul_20 = arg12_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_21, arg13_1);  mul_21 = arg13_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        mm_38 = torch.ops.aten.mm.default(mm_36, arg158_1);  mm_36 = arg158_1 = None
        add_30 = torch.ops.aten.add.Tensor(add_28, mm_38);  add_28 = mm_38 = None
        mul_50 = torch.ops.aten.mul.Tensor(add_30, convert_element_type);  add_30 = None
        mm_40 = torch.ops.aten.mm.default(mul_50, arg159_1);  mul_50 = arg159_1 = None
        permute_65 = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
        mm_43 = torch.ops.aten.mm.default(permute_65, convert_element_type);  permute_65 = convert_element_type = None
        return (mm_43,)
        
args = [((7616,), (1,), torch.float32, 'cuda'), ((7616,), (1,), torch.float32, 'cuda'), ((256, 7616), (7616, 1), torch.float32, 'cuda'), ((7616, 256), (256, 1), torch.float32, 'cuda'), ((8, 7616), (7616, 1), torch.float32, 'cuda'), ((8, 7616), (7616, 1), torch.float32, 'cuda'), ((8, 256), (256, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner