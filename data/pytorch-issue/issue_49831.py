import torch
import numpy as np

a = torch.zeros((1, 3, 10, 10))
b = torch.argmax(a, dim=1)
print (b.min(), b.max())

a = np.zeros((1, 3, 10, 10))
b = np.argmax(a, axis=1)
print(a.min(), b.max())