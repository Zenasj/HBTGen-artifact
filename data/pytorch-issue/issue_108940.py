import torch

x = torch.rand(100, 3, dtype=torch.bfloat16)
y = torch.rand(100, 3, dtype=torch.bfloat16)

torch.cross(x, y, dim=-1)  # Works

torch.cross(x.cuda(), y.cuda(), dim=-1)  # Fails