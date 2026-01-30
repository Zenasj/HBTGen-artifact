import torch
from torch import Tensor

@torch.no_grad()
@torch.compile(backend='aot_eager')
def compute_in_place_add(param: Tensor):
    view_param = param[:]
    view_param.add_(1.0) 
    return view_param
    
a = torch.tensor([1.0], requires_grad=True)
for _ in range(2):
    compute_in_place_add(a)

import torch
from torch import Tensor

@torch.no_grad()
def compute_in_place_add(param: Tensor):
    view_param = param[:]
    view_param.add_(1.0)
    return view_param

a = torch.tensor([1.0], requires_grad=True)
for _ in range(2):
    out = compute_in_place_add(a)
    # prints "RuntimeError: A view was created in no_grad mode and its base or another view of its base has been modified inplace with grad mode enabled. Given that this use case is ambiguous and error-prone, it is forbidden. You can clarify your code by moving both the view and the inplace either both inside the no_grad block (if you don't want the inplace to be tracked) or both outside (if you want the inplace to be tracked)."
    print(out.grad_fn)

import torch
from torch import Tensor

@torch.no_grad()
def compute_in_place_add(param: Tensor):
    view_param = param[:]
    view_param.add_(1.0)
    return view_param

a = torch.tensor([1.0], requires_grad=True)
for _ in range(2):
    compute_in_place_add(a.detach())