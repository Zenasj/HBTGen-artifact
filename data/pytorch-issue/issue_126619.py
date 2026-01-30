import torch.nn as nn

import logging

import torch

from kornia.filters import InRange

torch._logging.set_logs(dynamo=logging.DEBUG)
torch._dynamo.config.verbose = True


device = torch.device("cpu")
dtype = torch.float32
batch_size = 1

inpt = torch.rand(batch_size, 3, 5, 5, device=device, dtype=dtype)
op = InRange(lower=(0.2, 0.2, 0.2), upper=(0.6, 0.6, 0.6), return_mask=True)

op_optimized = torch.compile(op, backend="inductor")

op_optimized(inpt)

# TORCHDYNAMO_REPRO_AFTER="dynamo" python t.py

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
torch._dynamo.config.verbose = True








from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, L_input_ : torch.Tensor):
        input_1 = L_input_
        tensor = torch.tensor((0.2, 0.2, 0.2), device = device(type='cpu'), dtype = torch.float32)
        reshape = tensor.reshape(1, -1, 1, 1);  tensor = None
        lower = reshape.repeat(1, 1, 1, 1);  reshape = None
        tensor_1 = torch.tensor((0.6, 0.6, 0.6), device = device(type='cpu'), dtype = torch.float32)
        reshape_1 = tensor_1.reshape(1, -1, 1, 1);  tensor_1 = None
        upper = reshape_1.repeat(1, 1, 1, 1);  reshape_1 = None
        ge = input_1 >= lower;  lower = None
        le = input_1 <= upper;  input_1 = upper = None
        mask = torch.logical_and(ge, le);  ge = le = None
        all_1 = mask.all(dim = 1, keepdim = True);  mask = None
        output = all_1.to(torch.float32);  all_1 = None
        return (output,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('e2ab56dff6760f39c0f07bc28f2c2b46e05c4083', 300)
    reader.tensor(buf0, (1, 3, 5, 5), is_leaf=True)  # L_input_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify',
        save_dir='/tmp/kornia/torch_compile_debug/run_2024_05_18_11_39_38_826873-pid_26238/minifier/checkpoints', autocast=False, backend='inductor')

# TORCHDYNAMO_REPRO_AFTER="aot" python t.py


import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = True





isolate_fails_code_str = None



# torch version: 2.3.0
# torch cuda version: 12.1
# torch git version: 97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 3060 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', tensor([0.2000, 0.2000, 0.2000]))
        self.register_buffer('_tensor_constant1', tensor([0.6000, 0.6000, 0.6000]))

    
    
    def forward(self, arg0_1):
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        view = torch.ops.aten.view.default(lift_fresh_copy, [1, -1, 1, 1]);  lift_fresh_copy = None
        full_default = torch.ops.aten.full.default([1, 3, 1, 1], 0.20000000298023224, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        view_1 = torch.ops.aten.view.default(lift_fresh_copy_1, [1, -1, 1, 1]);  lift_fresh_copy_1 = None
        full_default_1 = torch.ops.aten.full.default([1, 3, 1, 1], 0.6000000238418579, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        ge = torch.ops.aten.ge.Tensor(arg0_1, full_default);  full_default = None
        le = torch.ops.aten.le.Tensor(arg0_1, full_default_1);  arg0_1 = full_default_1 = None
        logical_and = torch.ops.aten.logical_and.default(ge, le);  ge = le = None
        logical_not = torch.ops.aten.logical_not.default(logical_and);  logical_and = None
        any_1 = torch.ops.aten.any.dim(logical_not, 1, True);  logical_not = None
        logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(logical_not_1, torch.float32);  logical_not_1 = None
        return (convert_element_type,)
        
def load_args(reader):
    buf0 = reader.storage(None, 300)
    reader.tensor(buf0, (1, 3, 5, 5), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='minify', save_dir='/tmp/kornia/torch_compile_debug/run_2024_05_18_11_40_34_518846-pid_26379/minifier/checkpoints', tracing_mode='real', check_str=None)