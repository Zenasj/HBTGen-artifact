import numpy as np
import torch


@torch.compile
def f(x):
    return x + 1.0

for x in np.nditer(np.arange(3)):
    print(f(x))