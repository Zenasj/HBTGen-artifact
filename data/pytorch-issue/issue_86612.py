import torch.nn as nn

py
import torch
from functorch.compile import aot_function
from functorch.compile import make_boxed_compiler
from functorch.compile import get_aot_graph_name

def func(a):
    return torch.nn.functional.silu(a)

def fw_compiler(gm, args):
    print(get_aot_graph_name())
    gm.graph.print_tabular()
    return gm

aot_fn = aot_function(func, fw_compiler=make_boxed_compiler(fw_compiler))
a = torch.randn(3, 3, device="cuda", requires_grad=True)

aot_fn(a).backward(torch.ones_like(a))