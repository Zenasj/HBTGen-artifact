import torch.nn as nn
import random

py
import torch
torch._C._jit_set_nvfuser_single_node_mode(True)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')

    def forward(self, input):
        out = self.out
        out = torch.nn.functional.tanhshrink(out) # disable this, it will work
        return torch.logical_not(input, out=out)

input = torch.tensor([-1., 1., 1.], dtype=torch.float64, device='cuda')

m = M().to('cuda')
jit_m = torch.jit.trace(m, input)
print(jit_m(input))
# RuntimeError: tensor_inputs_to_check.size() INTERNAL ASSERT FAILED at "../torch/csrc/jit/codegen/cuda/graph_fuser.cpp":1558, please report a bug to PyTorch. CudaFusionGuard expects at least one tensor input

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
torch._C._jit_set_nvfuser_horizontal_mode(True)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        other = torch.randint(0, 2, [4, 1], dtype=torch.bool, device='cuda')
        alpha = 10

        self.other = other
        self.alpha = alpha

    def forward(self, input):
        other = self.other
        alpha = self.alpha

        other = torch.cos(other)
        other = torch.mul(other, torch.tensor(-12, dtype=torch.float32, device='cuda'))
        input = torch.nn.functional.relu(input)
        alpha = torch.sub(alpha, torch.tensor(-13, dtype=torch.float32, device='cuda'))
        res = input.add(other, alpha=alpha, )
        res = torch.sin(res)
        return res

fn = M().to('cuda')
input = torch.empty([4], dtype=torch.float32, memory_format=torch.contiguous_format, device='cuda')
input.uniform_(0, 31)

script_fn = torch.jit.script(fn)
for i in range(10):
    print(i)
    script_fn(input.clone())