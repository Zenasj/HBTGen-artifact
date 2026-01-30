import numpy as np
import torch

a = np.array([1], dtype=np.int64)
b = torch.tensor([1], dtype=torch.int64)
print((a[0] * b).dtype)