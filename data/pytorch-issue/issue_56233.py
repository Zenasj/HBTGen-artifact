import torch
from torch.testing._internal.common_utils import make_tensor
a = make_tensor((1, ), device='cpu', dtype=torch.float32) 
b = make_tensor((2, 2), device='cpu', dtype=torch.float32) 
c = make_tensor((2, 3), device='cpu', dtype=torch.float32)
a.addmm_(b, c)