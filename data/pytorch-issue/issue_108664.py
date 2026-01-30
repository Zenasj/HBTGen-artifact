import random

import torch
import numpy as np
np.random.seed(1234)
shape = (2,3,3)
a = np.random.rand(*shape)
b = np.random.rand(*shape)
a_tensor = torch.tensor(a, dtype=torch.float32)
b_tensor = torch.tensor(b, dtype=torch.float32)
out = torch.cross(a_tensor, b_tensor)
print("torch.cross's Output:\n", out)
out = torch.linalg.cross(a_tensor, b_tensor)
print("torch.linalg.cross's Output:\n", out)
print("Numpy's result: \n", np.cross(a, b))