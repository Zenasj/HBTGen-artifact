import torch

a = torch.rand(2, 2, 2)
a[a > 0.5] = 0
print(a[a > 0.5]) # 1d tensor

a = torch.rand(4, 4, 4)
mask = torch.rand(4) > 0.5
b = a[mask]
# equivalent to
c = a[mask.nonzero().squeeze(1)]