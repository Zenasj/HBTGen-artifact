import torch
a = torch.rand([3, 3])
b = torch.as_strided(a, [1, -1], [1, 1])
print(b.shape)
# torch.Size([1, -1])