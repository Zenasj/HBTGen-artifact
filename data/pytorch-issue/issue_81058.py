import torch
print(torch.__version__)
print(torch._C._PYBIND11_BUILD_ABI)

import torch
print(torch.__version__)
print(torch._C._PYBIND11_BUILD_ABI)
from functorch import vmap
x = torch.randn(2, 3, 5)
vmap(lambda x: x, out_dims=3)(x)