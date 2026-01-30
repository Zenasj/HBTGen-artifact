import random

import torch
import numpy as np

k = 31
shape = (k, 100) # shape of tensor and boolean mask
x = torch.rand(shape) # create random tensor to be indexed
mask = np.random.rand(*shape) > 0.5 # create random boolean array
assert x.shape == mask.shape # is true 

x[mask] # index the tensor