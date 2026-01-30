import torch.nn as nn

import torch

x = torch.rand((1, 5, 10))
model = torch.nn.modules.activation.MultiheadAttention(10, 1, bias=False, batch_first=True)
model.eval()
model(x, x, x)

import torch

x = torch.rand((1, 5, 10))
model = torch.nn.modules.activation.MultiheadAttention(10, num_heads=2, bias=False, batch_first=True)
model.eval()
model(x, x, x)