py
import torch
from torch.func import jacfwd, jacrev

torch.manual_seed(420)

x = torch.randn(3, 4)

def func(x):
    expected = torch.tensor([x[2, 0], x[1, 1], x[0, 2]])
    return expected

jacrev(func)(x)
# RuntimeError: unwrapped_count > 0 INTERNAL ASSERT FAILED 
# at "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/functorch/TensorWrapper.cpp":202, 
# please report a bug to PyTorch. Should have at least one dead wrapper

torch.stack(([x[2, 0], x[1, 1], x[0, 2]]))