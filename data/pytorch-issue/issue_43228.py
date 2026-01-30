import torch.nn.functional as F

import torch
from torch.nn import functional as F

A = torch.ones(2)
B = torch.zeros(2, requires_grad=True)
print(F.mse_loss(A, B, reduction='elementwise_mean'))

C = torch.zeros(2)
print(F.mse_loss(A, C, reduction='elementwise_mean'))