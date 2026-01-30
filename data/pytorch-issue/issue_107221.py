import torch

x = torch.randn(100, 100)
s1 = x.sum()
s2 = x.sum(0).sum(0)
print((s1 - s2).abs())
# tensor(3.0518e-05)