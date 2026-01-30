import torch
import torch.nn as nn

x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, requires_grad=True)
y = torch.nn.functional.elu_(x.clone(), alpha=-2)
grads = torch.tensor(torch.ones_like(y))
y.backward(grads)