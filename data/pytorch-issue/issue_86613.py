import torch.nn as nn

py
import torch
from functorch.compile import aot_function
from functorch.compile import make_boxed_compiler
import functorch
functorch.compile.config.use_functionalize = True # default

def func(a):
    return torch.nn.functional.silu(a)

def raise_error(*args):
    raise RuntimeError("Expected error")

d = {torch.ops.aten.silu_backward.default: raise_error}

def fw_compiler(gm, args):
    return gm

aot_fn = aot_function(func, fw_compiler=make_boxed_compiler(fw_compiler), decompositions=d)
a = torch.randn(3, 3, device="cuda", requires_grad=True)

try:
    aot_fn(a).backward(torch.ones_like(a))
    print("No error!")
except RuntimeError as e:
    print(e)