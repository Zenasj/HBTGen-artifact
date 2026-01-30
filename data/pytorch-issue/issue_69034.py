import torch
import torch.autograd.forward_ad as fwAD

x = torch.randn(3)
y = torch.randn(3)
tx = torch.randn(3)
ty= torch.randn(3)

with fwAD.dual_level():
  x_dual = fwAD.make_dual(x, tx)
  y_dual = fwAD.make_dual(y, ty)
  torch.maximum(x_dual, y_dual)