import torch

def f_non_contiguous(x, y, z):
    z_t = torch.ops.aten.t.default(z)
    return torch.ops.aten.addmm.default(x, y, z_t)

def f_contiguous(x, y, z):
    z_t = torch.ops.aten.t_copy.default(z)
    return torch.ops.aten.addmm.default(x, y, z_t)

x = torch.randn(256)
y = torch.randn(6, 960)
z = torch.randn(256, 960)

contiguous_result = f_contiguous(x, y, z)
non_contiguous_result = f_non_contiguous(x, y, z)

print(contiguous_result.is_contiguous())  # prints True 
print(non_contiguous_result.is_contiguous()) # print True 

print(torch.allclose(contiguous_result, non_contiguous_result)) # prints False
print(torch.allclose(contiguous_result.contiguous(), non_contiguous_result.contiguous())) # prints False

import torch

def f_non_contiguous(x, y, z):
    z_t = torch.ops.aten.t.default(z)
    return torch.ops.aten.addmm.default(x, y, z_t)

def f_contiguous(x, y, z):
    z_t = torch.ops.aten.t.default(z)
    return torch.ops.aten.addmm.default(x, y, z_t.contiguous())

x = torch.randn(2, 48)
y = torch.randn(2, 48)
z = torch.randn(48, 48)

contiguous_result = f_contiguous(x, y, z)
non_contiguous_result = f_non_contiguous(x, y, z)
print(torch.allclose(contiguous_result, non_contiguous_result)) # prints False