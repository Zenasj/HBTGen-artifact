import numpy as np

import torch
torch.set_default_device("cpu")
import torch._dynamo.config as cfg
cfg.numpy_ndarray_as_tensor = True

@torch.compile
def idx(Z, I):
    return Z[I]

Z = np.arange(12)
I = np.arange(12) % 2 == 0

print(idx(Z, I))