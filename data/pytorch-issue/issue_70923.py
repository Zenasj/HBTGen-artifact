import torch

condition = torch.tensor([False, True])
x = torch.ones(2, dtype=torch.float32)
y = torch.zeros(2, dtype=torch.float64)

torch.where(condition, x, y)