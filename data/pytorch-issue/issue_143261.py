import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

x = torch.tensor(2.0, requires_grad=True)
y = torch.pow(x,3)

import numpy as np

x = torch.tensor(2.0, requires_grad=True)
y = np.pow(x,3)
# > RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

import math

x = torch.tensor(2.0, requires_grad=True)
y = math.pow(x,3)

x = torch.tensor(2.0, requires_grad=True)
y = math.pow(x,3)
# > UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
# Consider using tensor.detach() first.