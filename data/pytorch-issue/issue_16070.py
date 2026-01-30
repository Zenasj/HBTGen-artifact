import torch
import numpy as np

A = torch.sparse_coo_tensor(np.vstack(([0], [0])), [1], size=(100, 10))
x = torch.ones(10)

y = torch.einsum("ij,j->i", (A,x))