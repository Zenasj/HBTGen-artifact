from typing import List
import torch
from torch import fx

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
torch._dynamo.reset()
compile_f1 = torch.compile(bar, backend=custom_backend)

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    raise ValueError
    gm.graph.print_tabular()
    return gm.forward

compile_f1 = torch.compile(bar, backend=custom_backend)
compile_f1(torch.randn(5), torch.randn(5))