import torch
x = torch.full((8,), 1e308, dtype=torch.float64)
y = torch.full((8,), 0.5, dtype=torch.float64)
print(torch.fmod(x, y))       # tensor([nan, nan, nan, nan, nan, nan, nan, nan], dtype=torch.float64)
print(torch.fmod(x[0], y[0])) # tensor(0., dtype=torch.float64)