import torch

x = torch.randn(1, 3456, 512, device="cuda")

def func(x, o_size, k_size, dilation, padding, stride):
    return torch.ops.aten.col2im(x, o_size, k_size, dilation, padding, stride)

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
decomp_table = get_decompositions({torch.ops.aten.col2im})
out = func(x, [512, 1], [9, 1], [1, 1], [4, 0], [1, 1])
print(out.shape)
gm = make_fx(func, decomposition_table=decomp_table)(x, [512, 1], [9, 1], [1, 1], [4, 0], [1, 1])
print(gm.graph)