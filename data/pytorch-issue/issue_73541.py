import torch.nn as nn

import torch
from torch import tensor
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
import functools

args = (
  torch.randn(3, 2, 5, device='cuda'),
  tensor([1.5041, 5.0046], device='cuda:0'),
  tensor([4.9961, 0.7634], device='cuda:0'),
  tensor([1.1864, 8.7912], device='cuda:0', requires_grad=True),
  tensor([-7.0369, -6.4842], device='cuda:0', requires_grad=True)
)
kwargs = {'training': False, 'momentum': -1.2}

fn = functools.partial(F.batch_norm, **kwargs)

tangents = tuple(torch.rand_like(x) for x in args)

with fwAD.dual_level():
  duals = [fwAD.make_dual(primal, tangent) for primal, tangent in zip(args, tangents)]
  fn(*duals)