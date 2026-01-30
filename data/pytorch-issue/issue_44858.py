import torch
a = torch.IntTensor([1, -1])
b = torch.IntTensor([2, 2])
print(torch.floor_divide(a, b))

tensor([0, 0], dtype=torch.int32)

tensor([0, -1], dtype=torch.int32)