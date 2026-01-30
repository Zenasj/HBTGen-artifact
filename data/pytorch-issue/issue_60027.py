import torch
import torch.nn as nn
x = torch.rand(40, 12, requires_grad=True)
A = nn.Linear(12, 1)
y = A(x)
g, = torch.autograd.grad(inputs=x, outputs=y, grad_outputs=torch.ones_like(y), create_graph=True)
g2, = torch.autograd.grad(inputs=g, outputs=y, grad_outputs=torch.ones_like(y), allow_unused=True)
# Will give None for g2