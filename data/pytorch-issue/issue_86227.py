import torch.nn as nn
import random

py
import torch

def fn(input):
    fn_res = torch.numel(input, )
    fn_res = torch.div(fn_res, torch.tensor(-3, dtype=torch.float32, device='cuda'))
    fn_res = torch.mul(fn_res, torch.tensor(-6, dtype=torch.float32, device='cuda'))
    return fn_res

input_tensor = torch.rand([1, 2, 3, 4, 5], dtype=torch.float32)

jit_fn = torch.jit.script(fn)
for i in range(5):
    print(i)
    jit_fn(input_tensor.clone().to('cuda'))

py
import torch
# Tensor Info
# {"name": "_input_tensor", "shape": [3], "dtype": "torch.float32"}
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input_tensor):
        fn_res = _input_tensor.dim()
        fn_res = torch.sub(fn_res, torch.tensor(-5, dtype=torch.float32, device='cuda'))
        fn_res = torch.sub(fn_res, torch.tensor(-7, dtype=torch.float32, device='cuda'))
        return fn_res

fn = M().to('cuda')

torch.random.manual_seed(54537)
input_tensor = torch.empty([3], dtype=torch.float32, memory_format=torch.contiguous_format)
input_tensor.uniform_(-32, 63)

jit_fn = torch.jit.script(fn)
for i in range(5):
    print(i)
    jit_fn(input_tensor.clone().to('cuda'))

py
import torch

torch._C._jit_set_nvfuser_single_node_mode(True)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input_tensor):
        _input_tensor = torch.nn.functional.tanhshrink(_input_tensor)
        _input_tensor = torch.cos(_input_tensor)
        fn_res = _input_tensor.element_size()
        fn_res = torch.sub(fn_res, torch.tensor(-7, dtype=torch.float32, device='cuda'))
        return fn_res

fn = M().to('cuda')

torch.random.manual_seed(32533)
input_tensor = torch.empty([1, 1, 3, 3], dtype=torch.float32, memory_format=torch.contiguous_format)
input_tensor.uniform_(-32, 127)

jit_fn = torch.jit.script(fn)
for i in range(5):
    print(i)
    jit_fn(input_tensor.clone().to('cuda'))

py
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input_tensor):
        fn_res = _input_tensor.get_device()
        fn_res = torch.mul(fn_res, torch.tensor(-4, dtype=torch.float32, device='cuda'))
        fn_res = torch.nn.functional.tanhshrink(fn_res)
        return fn_res

fn = M().to('cuda')

torch.random.manual_seed(23049)
input_tensor = torch.empty([1, 96, 1, 1], dtype=torch.float32, memory_format=torch.contiguous_format)
input_tensor.uniform_(-64, 3)

jit_fn = torch.jit.script(fn)
for i in range(5):
    print(i)
    jit_fn(input_tensor.clone().to('cuda'))