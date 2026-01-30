import torch.nn as nn

import torch

dtype = torch.bfloat16

a = torch.randn([1, 3, 1, 2, 2], device="cuda", dtype=dtype, requires_grad=True)
b = a.detach()[:, :, 0].clone().requires_grad_(True)

c = torch.nn.functional.interpolate(a, scale_factor=(1, 2, 2), mode="trilinear")
d = torch.nn.functional.interpolate(b, scale_factor=2, mode="bilinear")

torch.nn.functional.mse_loss(c, torch.zeros_like(c)).backward()
torch.nn.functional.mse_loss(d, torch.zeros_like(d)).backward()

print((a.grad[:, :, 0] - b.grad))