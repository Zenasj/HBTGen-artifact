import torch

x = torch.rand(4, 3, 3)
torch.norm(x, dim=(1,2))        # Works
torch.norm(x, p=1)              # Works
torch.norm(x, p=1, dim=(1,2))   # Breaks