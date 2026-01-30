import torch

assert not (torch.isnan(x).any() or torch.isinf(x).any())

assert not (torch.isnan(x).any() or x.eq(float('inf')).any() or x.eq(float('-inf')).any())