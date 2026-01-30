import torch

x = torch.randn((1))
x[0] = torch.iinfo(torch.int32).max
print(x)
print(x.to(torch.int32))

tensor([2.1475e+09])
tensor([-2147483648], dtype=torch.int32)

x[0] = torch.iinfo(torch.int32).max - 100

tensor([2.1475e+09])
tensor([2147483520], dtype=torch.int32)