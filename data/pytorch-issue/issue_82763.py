import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import enable_torch_dispatch_mode

with enable_torch_dispatch_mode(FakeTensorMode()):
    a = torch.rand([4])
    torch.nan_to_num(a, nan=3.0),