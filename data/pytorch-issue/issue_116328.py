import torch

x = torch.rand(2, 2)
reveal_type(x)
y = x + 1
reveal_type(y)