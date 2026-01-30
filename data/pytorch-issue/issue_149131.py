import torch
import torch.nn as nn

loss = nn.GaussianNLLLoss()
input = torch.randn(5, 2, requires_grad=True)
target = torch.randn(5, 2)
var = 1.0
output = loss(input, target, var)

import torch
import torch.nn.functional as F

input = torch.randn(5, 2, requires_grad=True)
target = torch.randn(5, 2)
var = 1.0
output = F.gaussian_nll_loss(input, target, var)