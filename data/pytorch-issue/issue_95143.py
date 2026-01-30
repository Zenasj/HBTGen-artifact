py
import torch

torch.manual_seed(0)

input = torch.randn(2, 3)
updates = torch.randn(2, 3)

def func(input, updates):
    index = torch.tensor([0, 2])
    result = input.index_add(0, index, updates)
    return result

# func(input, updates)
# IndexError: index out of range in self

torch.compile(func)(input, updates)
# crash: IOT instruction (core dumped)