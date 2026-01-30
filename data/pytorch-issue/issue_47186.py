import torch
class _T(torch.Tensor): ...
t = tensor([1.]).as_subclass(_T)
t.new([1,2])

t = tensor([1.])
t.new([1,2])