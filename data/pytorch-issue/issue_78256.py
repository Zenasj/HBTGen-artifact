import torch.nn as nn

import torch
import torch.nn.functional as F
from functools import partial


x = torch.randn(2, 3, 4, dtype=torch.cdouble, requires_grad=True)
t = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)

def f(u, v):
    return u - v
torch.autograd.gradgradcheck(f, (x, t), grad_outputs=torch.ones_like(f(x, t)), check_fwd_over_rev=True)

import torch
import torch.autograd.forward_ad as fwAD

x = torch.randn(2, 3, 4, dtype=torch.cdouble, requires_grad=True)
t = torch.randn(2, 3, 4, dtype=torch.cdouble, requires_grad=True).conj()

with fwAD.dual_level():
    dual = fwAD.make_dual(x, t)
    torch.real(dual)