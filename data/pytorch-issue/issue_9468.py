import torch
import numpy as np

tensor = torch.ones(2, requires_grad=True, dtype=torch.float32)
scalar = np.float64(2.0)
prod = tensor * scalar
print(prod, prod.requires_grad, prod.dtype)

tensor = torch.ones(2, requires_grad=True, dtype=torch.float32)
scalar = np.float64(2.0)
prod = scalar * tensor
print(prod, prod.requires_grad, prod.dtype)

tensor = torch.ones(2, requires_grad=True, dtype=torch.float32)
scalar = np.float32(2.0)
prod = tensor * scalar
print(prod, prod.requires_grad, prod.dtype)

tensor = torch.ones(2, requires_grad=True, dtype=torch.float32)
scalar = np.float32(2.0)
prod = scalar * tensor
print(prod, prod.requires_grad, prod.dtype)