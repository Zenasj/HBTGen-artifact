import torch.nn as nn

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config





isolate_fails_code_str = None



# torch version: 2.1.0
# torch cuda version: 11.8
# torch git version: 7bcf7da3a268b435777fe87c7794c382f444e86d


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 3060 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor(1.4660e+13))
        self.register_buffer('_tensor_constant1', tensor(1.4660e+13))

    
    
    def forward(self, arg0_1, arg1_1):
        _tensor_constant0 = self._tensor_constant0
        full_default = torch.ops.aten.full.default([], 14660154687488.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        expand = torch.ops.aten.expand.default(full_default, [1, 3, arg1_1]);  full_default = None
        slice_1 = torch.ops.aten.slice.Tensor(expand, 0, 0, 9223372036854775807);  expand = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze, [-1, 3, -1, -1]);  unsqueeze = None
        sub = arg1_1 - 1
        floordiv = sub // 2
        sub_1 = sub - floordiv;  sub = None
        reflection_pad2d = torch.ops.aten.reflection_pad2d.default(arg0_1, [floordiv, sub_1, 1, 1]);  floordiv = sub_1 = None
        view = torch.ops.aten.view.default(expand_1, [3, 1, 3, arg1_1]);  expand_1 = None
        sym_size = torch.ops.aten.sym_size(reflection_pad2d, 3)
        view_1 = torch.ops.aten.view.default(reflection_pad2d, [-1, 3, 6, sym_size]);  reflection_pad2d = sym_size = None
        convolution = torch.ops.aten.convolution.default(view_1, view, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3);  view_1 = view = None
        permute = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg0_1, 1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        mul = torch.ops.aten.mul.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        view_3 = torch.ops.aten.view.default(mul, [2, 9, 4, 4]);  mul = None
        _tensor_constant1 = self._tensor_constant1
        full_default_1 = torch.ops.aten.full.default([], 14660154687488.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        expand_2 = torch.ops.aten.expand.default(full_default_1, [1, 3, arg1_1]);  full_default_1 = None
        slice_2 = torch.ops.aten.slice.Tensor(expand_2, 0, 0, 9223372036854775807);  expand_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_3, [-1, 9, -1, -1]);  unsqueeze_3 = None
        sub_2 = arg1_1 - 1
        floordiv_1 = sub_2 // 2
        sub_3 = sub_2 - floordiv_1;  sub_2 = None
        reflection_pad2d_1 = torch.ops.aten.reflection_pad2d.default(view_3, [floordiv_1, sub_3, 1, 1]);  view_3 = floordiv_1 = sub_3 = None
        view_4 = torch.ops.aten.view.default(expand_3, [9, 1, 3, arg1_1]);  expand_3 = arg1_1 = None
        sym_size_1 = torch.ops.aten.sym_size(reflection_pad2d_1, 3)
        view_5 = torch.ops.aten.view.default(reflection_pad2d_1, [-1, 9, 6, sym_size_1]);  reflection_pad2d_1 = sym_size_1 = None
        convolution_1 = torch.ops.aten.convolution.default(view_5, view_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 9);  view_5 = view_4 = None
        permute_1 = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
        view_7 = torch.ops.aten.view.default(permute_1, [2, 4, 4, 3, 3]);  permute_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute, -2)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(permute, -1)
        mul_1 = torch.ops.aten.mul.Tensor(unsqueeze_4, unsqueeze_5);  unsqueeze_4 = unsqueeze_5 = None
        sub_4 = torch.ops.aten.sub.Tensor(view_7, mul_1);  view_7 = mul_1 = None
        return (permute, sub_4)
        
def load_args(reader):
    buf0 = reader.storage(None, 384)
    reader.tensor(buf0, (2, 3, 4, 4), is_leaf=True)  # arg0_1
    reader.symint(4)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify', save_dir='/tmp/kornia/torch_compile_debug/run_2023_10_08_18_15_23_052128-pid_10577/minifier/checkpoints', tracing_mode='symbolic', check_str=None)

import torch

from kornia.filters import GuidedBlur
from kornia.testing import assert_close

torch.set_float32_matmul_precision('high')

backend = 'inductor'
device = torch.device('cpu')
dtype = torch.float32
kernel_size = (5, 7)
subsample = 1

guide = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
inpt = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)

op = GuidedBlur(kernel_size, 0.1, subsample=subsample)
# will work fine
op_optimized = torch.compile(op, backend=backend)
out_a = op_optimized(guide, inpt)
out_b = op(guide, inpt)

assert_close(out_a, out_b)

kernel_size = 5
subsample = 2

guide = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)
inpt = torch.ones(2, 3, 8, 8, device=device, dtype=dtype)

op = GuidedBlur(kernel_size, 0.1, subsample=subsample)
# will fail if we already called it before
op_optimized = torch.compile(op, backend=backend) 
out_a = op_optimized(guide, inpt)
out_b = op(guide, inpt)

assert_close(out_a, out_b)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config





isolate_fails_code_str = None



# torch version: 2.1.0
# torch cuda version: 11.8
# torch git version: 7bcf7da3a268b435777fe87c7794c382f444e86d


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 3060 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor(2.5653e-21))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        _tensor_constant0 = self._tensor_constant0
        full_default = torch.ops.aten.full.default([], 2.5652997311834374e-21, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        expand = torch.ops.aten.expand.default(full_default, [1, arg3_1, arg4_1]);  full_default = None
        slice_1 = torch.ops.aten.slice.Tensor(expand, 0, 0, 9223372036854775807);  expand = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze, [-1, 3, -1, -1]);  unsqueeze = None
        sub = arg4_1 - 1
        floordiv = sub // 2
        sub_1 = sub - floordiv;  sub = None
        sub_2 = arg3_1 - 1
        floordiv_1 = sub_2 // 2
        sub_3 = sub_2 - floordiv_1;  sub_2 = None
        reflection_pad2d = torch.ops.aten.reflection_pad2d.default(arg2_1, [floordiv, sub_1, floordiv_1, sub_3]);  arg2_1 = floordiv = sub_1 = floordiv_1 = sub_3 = None
        view = torch.ops.aten.view.default(expand_1, [3, 1, arg3_1, arg4_1]);  expand_1 = arg3_1 = arg4_1 = None
        sym_size = torch.ops.aten.sym_size(reflection_pad2d, 2)
        sym_size_1 = torch.ops.aten.sym_size(reflection_pad2d, 3)
        view_1 = torch.ops.aten.view.default(reflection_pad2d, [-1, 3, sym_size, sym_size_1]);  reflection_pad2d = sym_size = sym_size_1 = None
        convolution = torch.ops.aten.convolution.default(view_1, view, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 3);  view_1 = view = None
        view_2 = torch.ops.aten.view.default(convolution, [2, 3, arg0_1, arg1_1]);  convolution = arg0_1 = arg1_1 = None
        return (view_2,)
        
def load_args(reader):
    reader.symint(8)  # arg0_1
    reader.symint(8)  # arg1_1
    buf0 = reader.storage(None, 1536)
    reader.tensor(buf0, (2, 3, 8, 8), is_leaf=True)  # arg2_1
    reader.symint(5)  # arg3_1
    reader.symint(7)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify', save_dir='/tmp/kornia/torch_compile_debug/run_2023_10_08_18_31_25_613568-pid_11702/minifier/checkpoints', tracing_mode='symbolic', check_str=None)