import torch

a, b, c, z = [torch.rand((3,2,2)) for _ in range(4)]

z[:] = torch.nan
torch.addcmul(c, a, b, out=z)
print(z.isnan().sum())   # -> tensor(0), NaNs overwritten, great

z[:] = torch.nan
torch.baddbmm(c, a, b, alpha=1, beta=0, out=z)
print(z.isnan().sum())   # -> tensor(12), `z` is all NaNs

z = c
z[1,1,1] = z[0,0,0] = torch.nan   # plant two NaNs
torch.baddbmm(c, a, b, alpha=1, beta=0, out=z)
print(z.isnan().sum())   # -> tensor(2)  two NaNs preserved