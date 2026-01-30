import torch

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
dataset = Data.TensorDataset(x, y)