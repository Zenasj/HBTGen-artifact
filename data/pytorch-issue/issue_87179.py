import torch
msk = torch.tensor([False])
b = torch.tensor([False])
print(torch.ops.aten.where.ScalarSelf(msk, True, b).dtype)