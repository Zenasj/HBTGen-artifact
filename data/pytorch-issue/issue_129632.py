import torch
x = torch.ones((int(90e9),), device='cuda', dtype=torch.uint8)
x.fill_(2.)