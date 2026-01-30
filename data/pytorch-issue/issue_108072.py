import torch.nn as nn

import torch
x = torch.Tensor([[[2.0, 2.0], [14.0, 14.0]], [[2.0, 2.0], [14.0, 14.0]]])
ln = torch.nn.LayerNorm(2, eps=1e-6)
for n,p in ln.named_parameters():
  torch.nn.init.ones_(p)
ln.eval()

y = ln(x)
print(y)

x * rstd_val + (-rstd_val * mean_val)