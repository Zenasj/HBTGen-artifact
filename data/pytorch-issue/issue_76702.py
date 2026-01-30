import torch

x = torch.rand(10, 20, 30, 40)
weight = torch.rand(30, 40, 50)

print(torch.einsum("btgi,gih->btgh", x, weight).shape)
# torch.Size([10, 20, 30, 50])

print(torch.matmul(x, weight[None, None, ...]).shape)
# RuntimeError: The size of tensor a (20) must match the size of tensor b (30) at non-singleton dimension 2