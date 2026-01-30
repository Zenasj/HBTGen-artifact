import torch
import numpy as np

@torch.compile(fullgraph=False)
def fn():
    return np.asarray(["L", "U"])