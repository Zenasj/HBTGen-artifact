import torch
import torch.nn as nn

py
y = torch.nn.functional.linear(x, weight, bias)
y = y.pow(2)
# first order derivative
y__x, = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True)
# first order derivative
y__x__x, = torch.autograd.grad(y__x, x, grad_outputs=grad_outputs, create_graph=True)