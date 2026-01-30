import torch.nn as nn

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
torch._dynamo.config.assume_static_by_default = False
from torch.nn import *

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, view_22, getitem_51):
        expand_as_16 = view_22.expand_as(getitem_51);  view_22 = getitem_51 = None
        return (expand_as_16,)

mod = Repro()

def load_args(reader):
    buf0 = reader.storage('d4bfc3c81078c64c193b38e7b2189c50a5687e7a', 16, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 1), dtype=torch.int64, storage_offset=1, is_leaf=True)  # view_22
    buf1 = reader.storage('549a91c2aa84c73c8e9be35307048f20aeb4fd99', 128, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 32), requires_grad=True)  # getitem_51
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=True, command='run',
        save_dir='/home/akihiro/work/github.com/kumo-ai/some-repo/checkpoints', autocast=False, backend='inductor')

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._inductor.config.generate_intermediate_hooks = True
isolate_fails_code_str = None

# torch version: 2.1.0+cu118
# torch cuda version: 11.8
# torch git version: 7bcf7da3a268b435777fe87c7794c382f444e86d

# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2021 NVIDIA Corporation 
# Built on Thu_Nov_18_09:45:30_PST_2021 
# Cuda compilation tools, release 11.5, V11.5.119 
# Build cuda_11.5.r11.5/compiler.30672275_0 

# GPU Hardware Info: 
# Tesla T4 : 1 

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, arg0_1, arg3_1, convert_element_type, convert_element_type_1):
        scatter_add = torch.ops.aten.scatter_add.default(convert_element_type, 0, arg0_1, convert_element_type_1);  convert_element_type = arg0_1 = convert_element_type_1 = None
        div = torch.ops.aten.div.Tensor(arg3_1, scatter_add);  arg3_1 = scatter_add = None
        return (div,)
        
def load_args(reader):
    buf0 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 32), (1, 0), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage('9489f89ab0bd1a88139da1c856558f8fb29dadea', 384, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 32), (96, 1), storage_offset=64, is_leaf=True)  # arg3_1
    buf2 = reader.storage('da7a883c0f27c7ea8bc5974535fd0a6690fcf06c', 128, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 32), is_leaf=True)  # convert_element_type
    buf3 = reader.storage('26fa883ccca7c7eae74597450dfd0a66187cf06c', 128, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, 32), is_leaf=True)  # convert_element_type_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=True, command='run', save_dir='/home/akihiro/work/github.com/kumo-ai/kumo/torch_compile_debug/run_2023_11_26_23_49_34_096419-pid_692919/minifier/checkpoints', tracing_mode='real', check_str=None)