import torch

mask1 = torch.zeros(3)
mask1[True][True]=1.
print(mask1)