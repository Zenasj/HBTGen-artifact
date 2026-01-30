import torch

x=torch.arange(4.0, device='mps').reshape(2, 2)
y=torch.empty(2, 2, device='mps').t()
print(torch.neg(x, out=y))