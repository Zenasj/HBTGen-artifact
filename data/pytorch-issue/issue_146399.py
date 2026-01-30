import random

import torch
import numpy as np
x = torch.tensor(np.random.randn(10, 10)) # input with unexpected size
vec1 = torch.tensor(np.random.randn(3))
vec2 = torch.tensor(np.random.randn(3))
out = torch.addr(x, vec1, vec2, beta=0) # expected behavior: ignore input and just calc between vec1 & vec2