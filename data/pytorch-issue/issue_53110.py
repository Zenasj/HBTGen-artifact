import torch.nn as nn

py
import torch
import torch.nn.functional as F

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3, 1))

# Raises RuntimeError
loss = F.cross_entropy(input, target)