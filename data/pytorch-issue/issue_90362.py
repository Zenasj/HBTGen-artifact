import torch
from typing import List
import torch._dynamo

def toy(a, b):
    x = a / (torch.abs(a) + 1)
    if (b.sum() < 0):
        b = b* -1
    return x*b
compiled_toy = torch.compile(toy)
print(compiled_toy(torch.randn(4), torch.randn(4)))