import torch
from torch.distributions.categorical import Categorical
@torch.compile(fullgraph=True, mode='reduce-overhead')
def func():
    sample = Categorical(torch.rand(10, 5)).sample()

import torch

@torch.compile(fullgraph=True, mode='reduce-overhead')
def num():
    a = torch.randn(1, 2, 3, 4, 5)
    return torch.numel(a)

print("--no of elements: ",num())