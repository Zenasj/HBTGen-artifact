py
import torch
b = torch.tensor([2., 4.])

def func(b):
    a = torch.tensor([[1., 2.],
                        [4., 5.],
                        [7., 8.]])
    a.index_add_(0, torch.tensor([0, 2]), b)
    return a

ans = func(b.clone().requires_grad_())
ans.sum().backward()