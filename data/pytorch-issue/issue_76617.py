import torch
input_tensor = torch.rand([5, 5, 5], dtype=torch.complex128)
dimension = 2
size = -1
step = 2
res = input_tensor.unfold(dimension, size, step)
print(res.shape)
# torch.Size([5, 5, 4, -1])