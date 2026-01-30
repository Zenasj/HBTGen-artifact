import torch
x = torch.rand(6,6)
torch.linalg.eig(x)  # Okay
x[0, 4] = float('nan')
torch.linalg.eig(x)