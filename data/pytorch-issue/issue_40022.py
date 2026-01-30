import torch

a = torch.randn((32767,1))
b = torch.argmin(a,dim=1)
torch.max(b)

a = torch.randn((32767,1))
b = torch.argmin(a,dim=1)
torch.max(b)