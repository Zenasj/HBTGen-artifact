import torch
tensor = torch.randn(10).reshape(2, 5)
with torch.autograd.profiler.profile() as prof:
    torch.t(tensor)
print(prof)

import torch
tensor = torch.randn(10).reshape(2, 5)
with torch.autograd.profiler.profile() as prof:
    torch.transpose(tensor, 0, 1)
print(prof)