from typing import List
import torch
from torch import _dynamo as torchdynamo
@torch.compile(backend="eager")
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))