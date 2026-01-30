import torch

x = torch.rand(2, 2)
reveal_type(x)
y = x.mean()
reveal_type(y)