import numpy as np
import torch

def fn(x):
    if issubclass(x.__class__, np.ndarray):
        return 1
    return 0

opt_fn = torch.compile(fn, backend="eager")
opt_fn(np.ones([3, 3]))