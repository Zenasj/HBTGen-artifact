import cupy as cp
import torch
a = cp.asarray([1 + 1j*4, 2 + 1j*3, 5 - 1j*2])
a_tensor = torch.as_tensor(a, device='cuda:0')
a_tensor.__cuda_array_interface__['data']

import torch
x = torch.randn(2,2, dtype=torch.cfloat, device='cuda:0')
x.__cuda_array_interface__['data']