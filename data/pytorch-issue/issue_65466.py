import torch.nn as nn

import torch
from torch.autograd import gradgradcheck
from torch.nn.functional import softmax, log_softmax, kl_div

torch.manual_seed(0)
# input should be log probabilities
input = log_softmax(torch.randn(3, dtype=torch.float64, requires_grad=True))
# target should be probabilities
target = softmax(torch.randn_like(input, requires_grad=True))

gradgradcheck(kl_div, inputs=(input, target))

import torch
# !pip install torchviz
import torchviz


a = torch.rand(1, 10, requires_grad=True)
t = torch.rand(1, 10, requires_grad=True)

loss = torch.kl_div(a, t).sum()

ga, gt = torch.autograd.grad(loss, (a, t), create_graph=True)

torchviz.make_dot((loss, ga, gt), params={k:v for k,v in locals().items() if isinstance(v, torch.Tensor)})