import torch

x = torch.tensor([[1.]])
y = torch.tensor([[0.]])

y[:, 0] = x[:, 0]