import torch

x = torch.randn(2, 3)
xz = x[:]
assert x._version == xz._version
x.add_(1)
assert x._version == xz._version  # We can confirm that `x` and `xz` have the same version counter

x.data = torch.randn(3, 4)
x.add_(1)
assert x._version == xz._version  # We can confirm that `x` and `xz` still have the same version counter