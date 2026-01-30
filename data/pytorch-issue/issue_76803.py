import torch

a = torch.randn((2, 2), dtype=torch.float)
b = torch.tensor(1, dtype=torch.cdouble)
(a + b).dtype