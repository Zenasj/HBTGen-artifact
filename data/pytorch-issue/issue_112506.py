from typing import List
import torch
from torch import _dynamo as torchdynamo
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    print([x.__class__ for x in example_inputs])
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

@torch.compile(backend=my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    b = b * -1
    y = x * b
    z = y.sum()
    z.backward()
for _ in range(100):
    input = torch.randn(10, requires_grad=True), torch.randn(10, requires_grad=True)
    toy_example(*input)