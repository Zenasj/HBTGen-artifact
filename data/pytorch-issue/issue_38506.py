import torch.nn as nn

import torch

inp = torch.randn(12, requires_grad=True)
model = nn.Sequential(
    nn.Linear(12, 1)
)
out = model(inp)
grad = torch.autograd.grad(out, inp, create_graph=True)[0]
grad2 = torch.autograd.grad(grad.sum(), inp)[0]

import torch

inp = torch.randn(12, requires_grad=True)
model = nn.Sequential(
    nn.Linear(12, 1),
    nn.Sigmoid()
)
out = model(inp)
grad = torch.autograd.grad(out, inp, create_graph=True)[0]
grad2 = torch.autograd.grad(grad.sum(), inp)[0]

import torch

inp = torch.randn(3, requires_grad=True)
model = lambda x: (x**2).sum()
out = model(inp)
grad = torch.autograd.grad(out, inp, create_graph=True)[0]
grad2 = torch.autograd.grad(grad.sum(), inp, create_graph=True)[0]
grad3 = torch.autograd.grad(grad2.sum(), inp, create_graph=True)[0]
# The grad3 is now a tensor of 0