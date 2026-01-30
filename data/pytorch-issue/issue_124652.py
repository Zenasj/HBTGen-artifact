import torch
from torch import Tensor
from typing import Optional
import torch._prims_common as utils
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True


@torch.compile(fullgraph=True)
def f(x, w):
    a, b = x.tolist()
    ta = torch.randn(a)
    tb = torch.randn(b)
    pa = ta * w  # make it require gradients
    pb = tb * w
    r = torch.cat([pa, pb])
    #r = r.sigmoid()
    return r.sum()


x = torch.tensor([4, 9])
w = torch.randn(1, requires_grad=True)
f(x, w).backward()