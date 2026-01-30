import torch

x = torch.empty(1, device='meta')

print(x[0:1:2])
print(torch.ops.prims.slice.default(x, [0], [1], [2]))