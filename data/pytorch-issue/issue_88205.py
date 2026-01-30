import torch

x = torch.zeros(2)
with torch.no_grad():
    y = x.view(2)
    y.requires_grad = True
z = y.view(2)

z /= 8.0

import torch

x = torch.zeros(2, device='meta')
with torch.no_grad():
    y = x.view(2)
    y.requires_grad = True
z = y.view(2)

z /= 8.0