import torch
import numpy as np

@torch.compile
def fn(x):
    return x ** 2

fn(np.arange(8))