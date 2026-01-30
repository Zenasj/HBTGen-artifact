import torch
import numpy as np

t = torch.empty((2, 0, 4))
a = t.numpy()

t_res = torch.kthvalue(t, k=1, dim=2).values
a_res = np.partition(a, 1, axis=2)

assert t_res.shape == a_res.shape