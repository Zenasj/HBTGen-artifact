import torch
import numpy as np

a = np.zeros((2, 4, 3))
b = np.ones((2, 4), dtype = np.bool)
c = torch.ones((2, 4)).eq(1)

print(a[b].shape)
# (8, 3)

print(a[c].shape)
# (2, 4, 4, 3)

a[c.numpy().astype(np.bool)]