import torch

temp_tensor = torch.randn((), dtype=torch.half)

self = temp_tensor.to_sparse()
self.permute(())