import torch

base = torch.randn(1)
temp_tensor = torch.quantize_per_tensor(base, 0.1, 10, dtype=torch.qint32)

temp_tensor.any(None, False)