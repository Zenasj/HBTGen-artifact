import torch.nn as nn

import torch
x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, requires_grad=True)
y = torch.nn.functional.elu(x, alpha=-2)
print(y)
grads = torch.tensor(torch.ones_like(y))
y.backward(grads)
print(x.grad)