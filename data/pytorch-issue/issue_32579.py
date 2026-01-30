import torch
x = torch.ones(3, 5)
mask = torch.tensor([0,1,1], dtype=torch.bool)
x1 = x.clone()
x2 = x.clone()
x1[mask] = 0
x2[mask][:, 0] = 0

print(x1)
print(x2)

import torch
x = torch.randn(3, 5)
mask0 = torch.tensor([0,1,1], dtype=torch.bool)
mask1 = torch.tensor([1,0,0,0,0], dtype=torch.bool)
x[mask0,mask1] = 0
print(x)

x[1:,0] = 0