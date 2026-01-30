py
import torch
from torch.func import jacrev

torch.manual_seed(420)

x = torch.rand(1000000)

def func(x):
    y = torch.tensor([1,1,1,1,1,1,1,1,1,1])
    z = torch.take(x, y)
    return z

x_clone = x.clone().requires_grad_()
func(x_clone).sum().backward()
print(x_clone.grad)
# tensor([ 0., 10.,  0.,  ...,  0.,  0.,  0.])

jacrev(func)(x)
# RuntimeError: vmap: aten::put_(self, *extra_args) is not possible because there exists a Tensor `other` 
# in extra_args that has more elements than `self`. 
# This happened due to `other` being vmapped over but `self` not being vmapped over at level 1. 
# Please try to use out-of-place operators instead of aten::put_. 
# If said operator is being called inside the PyTorch framework, please file a bug report instead.