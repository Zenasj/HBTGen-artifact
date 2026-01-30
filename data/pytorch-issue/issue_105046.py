import numpy as np
import torch
# To run on CUDA, change "cpu" to "cuda" below.
device = "cuda"
torch.set_default_device(device)
print(f"{device = }")
import torch._dynamo.config as cfg
cfg.numpy_ndarray_as_tensor = True
x = np.arange(10)
@torch.compile
def fn(x):
    return x**2
print(">>> f{device =} >>> ", fn(x))

import torch
# To run on CUDA, change "cpu" to "cuda" below.

device = "cuda"
x = torch.arange(10)
torch.set_default_device(device)

@torch.compile
def noop(x):
    return x**2
print(f">>> f{device =} >>> ", noop(x))