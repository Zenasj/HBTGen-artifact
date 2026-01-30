import torch

"""Compute the scalar-valued second-order derivative of f(x, y) w.r.t. x.

Use Hessian-vector products (double-backward pass) in combination with
full_backward_hook.
"""

from torch import ones_like, rand, rand_like
from torch.autograd import grad
from torch.nn import MSELoss

x = rand(1)
x.requires_grad_(True)
y = rand_like(x)

# without hook (working in 1.12.1 and 1.13.0)
lossfunc = MSELoss()
f = lossfunc(x, y)

(gradx_f,) = grad(f, x, create_graph=True)
(gradxgradx_f,) = grad(gradx_f @ ones_like(x), x)

# with hook (working in 1.12.1 and broken in 1.13.0
lossfunc = MSELoss()


def hook(module, grad_input, grad_output):
    print("This is a test hook")


lossfunc.register_full_backward_hook(hook)

f = lossfunc(x, y)

# this line triggers the backward hook as expected
(gradx_f,) = grad(f, x, create_graph=True)
# the double-backward with hook crashes in 1.13, but used to work before
try:
    (gradxgradx_f,) = grad(gradx_f @ ones_like(x), x)
except RuntimeError as e:
    print(f"Caught RuntimeError: {e}")