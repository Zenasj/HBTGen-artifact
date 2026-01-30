import torch
import torch.nn as nn
import torch.nn.functional as F

B = 4
T = 1
device = "mps"

lin = nn.Linear(10, 256).to(device)
x = torch.rand([B, T, 10], device=device, requires_grad=True)
x = lin(x)
# x = lin(x.reshape(B * T, -1)).reshape(B, T, -1)  # does not error

assert x.shape == (B, T, 256)

cls_token = torch.rand([1, 256], device=device, requires_grad=True).repeat(B, 1, 1)
x = torch.cat([cls_token, x], dim=1)

x = x.transpose(0, 1) # if this line is commented out, does not error either

loss = F.mse_loss(x, torch.rand_like(x))

loss.backward()