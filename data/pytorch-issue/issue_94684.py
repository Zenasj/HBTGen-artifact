py
import torch
from torch.autograd.functional import jacobian
from torch.func import jacrev, jacfwd

torch.manual_seed(420)

x = torch.sparse.FloatTensor(5, 5)
y = torch.randn(5, 5)
z = torch.randn(5, 5)

def func(x, y, z):
    x.addmm_(y, z)
    return x

func(x, y, z)
# RuntimeError: r.layout() == kStrided 
# INTERNAL ASSERT FAILED at "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/native/sparse/SparseTensorMath.cpp":1253, 
# please report a bug to PyTorch. addmm_sparse_dense: expected strided result tensor, got tensor with layout Sparse