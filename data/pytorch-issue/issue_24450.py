py
import torch
xs = torch.randn(256,16,16)
[x.eig() for x in xs]

py
import torch
xs = torch.randn(256,16,16).unbind(0)
[x.eig() for x in xs]