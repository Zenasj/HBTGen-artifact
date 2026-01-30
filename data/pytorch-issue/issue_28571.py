import torch

mask = torch.BoolTensor(2, 2)
print(mask.shape, mask.dtype)
# (torch.Size([2, 2]), torch.bool)