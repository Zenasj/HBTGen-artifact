import random

import torch
import numpy as np
class A:
    def run(self):
        print("HELLO")
        tinput = torch.tensor(np.random.rand(100, 15))
        print("GOODBYE")
        return 5

def copy_without_deadlock(xs: np.ndarray) -> torch.Tensor:
    out = torch.zeros(xs.shape)
    for index in np.ndindex(*xs.shape):
        out[index] = float(xs[index])
    return out