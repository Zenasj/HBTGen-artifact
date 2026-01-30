import torch

py
x = torch.rand([1, 2, 3], dtype=torch.float64, requires_grad=True)

def func(x):
    x.retain_grad()
    return x

torch.autograd.gradcheck(func, (x,), check_forward_ad=True, check_backward_ad=False)
# RuntimeError: can't retain_grad on Tensor that has requires_grad=False

py
x = torch.rand([1, 2, 3], dtype=torch.float64, requires_grad=True)

def func(x):
    x.retain_grad()
    return x

torch.autograd.gradcheck(func, (x,), check_forward_ad=False, check_backward_ad=True)
# succeed without error