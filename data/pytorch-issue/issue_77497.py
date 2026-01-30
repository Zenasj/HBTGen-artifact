# add in the file torch/autograd/gradcheck.py
import torch

def fn(input, min):
    fn_res = torch.clip(input, min=min, )
    return fn_res

input = torch.rand([5], dtype=torch.float64, requires_grad=True)
min = torch.rand([], dtype=torch.float64, requires_grad=True)
output = fn(input, min)
output = _as_tuple(output)
fwd_jacobians = _get_analytical_jacobian_forward_ad(fn, (input, min), output)
print(fwd_jacobians)
# ((tensor([[1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 1.]], dtype=torch.float64),), (tensor([[0., 0., 0., 0., 0.]], dtype=torch.float64),))

import torch

def fn(input, min):
    fn_res = torch.clip(input, min=min, )
    return fn_res

input = torch.rand([5], dtype=torch.float64, requires_grad=True)
min = torch.rand([], dtype=torch.float64, requires_grad=True)

torch.autograd.gradcheck(fn, (input, min), check_backward_ad=False, check_forward_ad=True)
# NotImplementedError: Trying to use forward AD with clamp that does not support it.

import torch

def fn(input, min):
    fn_res = torch.clip(input, min=min, )
    return fn_res

input_tensor = torch.tensor(0.1, dtype=torch.float64)
min_tensor = torch.tensor(0.2, dtype=torch.float64)

input = input_tensor.clone().detach().requires_grad_()
min = min_tensor.clone().detach().requires_grad_()
fn(input, min=min).backward()
print(input.grad, min.grad)
# tensor(0., dtype=torch.float64) tensor(1., dtype=torch.float64)

from gradcheck import _get_analytical_jacobian_forward_ad
input = input_tensor.clone().detach().requires_grad_()
min = min_tensor.clone().detach().requires_grad_()
output = fn(input, min=min)
fwd_jacobians = _get_analytical_jacobian_forward_ad(fn, (input, min,), (output,))
print(fwd_jacobians)
# ((tensor([[0.]], dtype=torch.float64),), (tensor([[0.]], dtype=torch.float64),))