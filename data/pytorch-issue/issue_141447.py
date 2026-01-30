import torch

self = torch.full((9, 7, 9, 9,), 1e+13, dtype=torch.double)
padding = [-1, -1]
torch.ops.aten.reflection_pad1d(self, padding)