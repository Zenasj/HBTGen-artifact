import torch
import torch.nn.functional as F

x = torch.randn(2, 2, 2).to('cuda')
pred, idx = F.max_pool2d_with_indices(x, 2)
print(pred.shape) # torch.Size([2, 1, 1])
print(idx.shape) # torch.Size([1, 2, 1, 1]) which is inconsistent

x = torch.randn(2, 2, 2).to('cpu')
pred, idx = F.max_pool2d_with_indices(x, 2)
print(pred.shape) # torch.Size([2, 1, 1])
print(idx.shape) # torch.Size([2, 1, 1]), expected