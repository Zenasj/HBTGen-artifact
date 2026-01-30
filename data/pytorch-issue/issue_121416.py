import torch.nn as nn

import torch
from torch import nn

L = nn.MSELoss(reduction="none")  # Same result with SmoothL1Loss or HuberLoss

x = torch.zeros(2, requires_grad=True)
y = torch.Tensor([0, torch.nan])
mask = torch.logical_not(torch.isnan(y))

loss = L(x[mask], y[mask]).sum()
loss.backward()
print(x.grad)  # tensor([0., 0.]) -> expected result

loss = L(x, y)[mask].sum()
loss.backward()
print(x.grad)  # tensor([0., nan]) -> unexpected