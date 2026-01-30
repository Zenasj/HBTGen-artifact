py
import torch
import torch.nn as nn

net = nn.Sequential(
  nn.InstanceNorm2d(1),
  nn.ReLU(True)
)

x = torch.randn(1, 1, 1, 1, requires_grad=True)

g, = torch.autograd.grad(net(x).pow(2), [x], create_graph=True)
torch.autograd.grad(g.sum(), [x])

py
import torch
import torch.nn as nn

net = nn.Sequential(
  nn.InstanceNorm2d(3),
  nn.ReLU(True),
  nn.MaxPool2d(2)
)

x = torch.randn(1, 3, 3, 3, requires_grad=True)

g, = torch.autograd.grad(net(x).sum(), [x], create_graph=True)
torch.autograd.grad(g.sum(), [x])