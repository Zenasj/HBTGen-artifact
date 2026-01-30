import numpy as np

import torch
torch.set_default_device("cpu")

import torch._dynamo.config as cfg
cfg.numpy_ndarray_as_tensor = True

y_glob = np.array([[1, 2, 3], [4, 5, 6]])

@torch.compile(backend='eager', fullgraph=True)
def av(x):
    return abs(x)

print(av(torch.randn(2, 2)))
# print(av(y_glob))