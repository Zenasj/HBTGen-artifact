import torch
from typing import List

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def custom_backend(gm: torch.fx.GraphModule,  example_inputs: List[torch.Tensor]):
    torch.save(example_inputs, "./example_inputs.zip", pickle_protocol=4)

    return gm.forward

compiled_bar = torch.compile(bar, backend=custom_backend)

compiled_bar(torch.randn(2), torch.randn(2))
compiled_bar(torch.randn(4), torch.randn(4))

[tensor([1.6226, 0.0726]), tensor([-0.5025, -0.0696])]
[tensor([-0.5025, -0.0696]), tensor([0.6187, 0.0677])]
[s0, tensor([-0.4205, -0.7103, -0.8642, -1.0742]), s1, tensor([ 0.1903, -0.7546,  0.4817, -1.1801])]