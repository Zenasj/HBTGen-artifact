import torch

base = torch.randn(2,2)
self = torch.quantize_per_tensor(base, 0.1, 10, torch.quint4x2)
base2 = torch.randn(2,2)
other = torch.quantize_per_tensor(base2, 0.1, 10, torch.quint4x2)
self.equal(other)