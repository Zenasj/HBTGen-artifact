import torch
import numpy as np

t = torch.rand((1, 1, 0))
a = t.numpy()

t_norm = torch.linalg.norm(t, ord=2, dim=(0, 1), keepdim=False)
a_norm = np.linalg.norm(a, ord=2, axis=(0, 1), keepdims=False)

assert t_norm.shape == a_norm.shape