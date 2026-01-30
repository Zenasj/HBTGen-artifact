import torch
device="cpu"
x = torch.rand(4, device=device)
y = torch.rand(3, 0, device=device)
z = torch.rand(0, 4, device=device)
torch.addmm(x, y, z)

import torch
device="cuda"
x = torch.rand(4, device=device)
y = torch.rand(3, 0, device=device)
z = torch.rand(0, 4, device=device)
torch.addmm(x, y, z)