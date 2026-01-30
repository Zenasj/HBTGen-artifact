import random

import torch
import numpy as np


@torch.compile
def test_optimized(input, mat, vec):
    return torch.addmv(input, mat, vec)

def test(input, mat, vec):
    return torch.addmv(input, mat, vec)

input = torch.tensor([2], dtype=torch.int32)
mat = torch.tensor(np.random.randn(0, 0), dtype=torch.int32)
vec = torch.tensor([])
origin_out = test(input, mat, vec)
optimized_out = test_optimized(input, mat, vec)
print(origin_out)  # tensor([2.])
print(optimized_out)  # tensor([])