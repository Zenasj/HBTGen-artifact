import torch
from torch._dynamo.backends.common import aot_autograd
from torch._decomp import core_aten_decompositions

def inner_compiler(fx_module: torch.fx.GraphModule, example_inputs):
    print(fx_module.code)
    return fx_module

aot_backend = aot_autograd(fw_compiler=inner_compiler, decompositions=core_aten_decompositions())

def fn(i1, i2):
    return torch.diag(i1, i2)
torch._dynamo.reset()
c = torch.compile(fn, backend=aot_backend)
x = torch.rand((8,8), dtype=torch.float)
y = 0
c(x, y)

def forward(self, arg0_1):
    diagonal_copy = torch.ops.aten.diagonal_copy.default(arg0_1);  arg0_1 = None
    return (diagonal_copy,)