import torch.nn as nn

py
import torch
from torch.func import jacrev, jacfwd

torch.manual_seed(420)

p = torch.nn.Parameter(torch.ones(2, 3))

def func(p):
    p.data = torch.ones(1, 2, 3) * 2
    return p

jacrev(func)(p)
# RuntimeError: false INTERNAL ASSERT FAILED 
# at "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/functorch/TensorWrapper.cpp":137, 
# please report a bug to PyTorch. NYI

py
import torch
from torch.func import jacrev, jacfwd

torch.manual_seed(420)

x = torch.randn(2, 3, requires_grad=True)

def func(x):
    m = torch.nn.LazyBatchNorm1d(3)
    out = m(x)
    return out

jacrev(func)(x)
# RuntimeError: false INTERNAL ASSERT FAILED 
# at "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/functorch/TensorWrapper.cpp":137, 
# please report a bug to PyTorch. NYI