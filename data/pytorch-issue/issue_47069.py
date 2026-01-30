import torch
class _T(torch.Tensor): ...
x = torch.tensor([1,2]).as_subclass(_T)
x.new_ones((2,3))

tensor([[1, 1, 1],
        [1, 1, 1]])

import torch
class _T(torch.Tensor): ...
x = torch.tensor([1,2]).as_subclass(_T)
x.new((2,3))

import torch
class _T(torch.Tensor): ...
x = torch.tensor([1,2]).as_subclass(_T)
torch.max(x, dim=0)