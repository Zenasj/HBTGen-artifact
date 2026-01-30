import torch

a = torch.tensor([1.,2.])
a.dtype
# torch.float32

torch.set_default_dtype(torch.float64)

a = torch.tensor([1.,2.])
a.dtype
# torch.float64

torch.set_default_dtype(torch.float64)

a = torch.tensor([1])
a.dtype
# torch.int64