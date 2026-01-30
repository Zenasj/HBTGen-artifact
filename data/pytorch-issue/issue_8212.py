import torch
c = torch.arange(101)[None,:].repeat(50, 1)
c /= c[:,100:]
print(c.max())