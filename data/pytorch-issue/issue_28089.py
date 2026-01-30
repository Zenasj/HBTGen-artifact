py
import torch
print(torch.__version__)
x = torch.randn(1, 3, 4, 4)
print(x)
x = torch.var(x, dim=(2, 3), keepdim=True)
print(x)
print(x.size())