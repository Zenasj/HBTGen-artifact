import torch
import numpy as np

@torch.compile
def f(a, b):
    return a + b

with torch.device("mps"):
    f(np.array([1]), np.array([2]))