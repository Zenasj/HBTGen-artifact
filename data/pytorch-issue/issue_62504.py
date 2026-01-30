import torch

x = torch.randn(3, 3, requires_grad=True)
torch.signbit(x)