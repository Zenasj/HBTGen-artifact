import torch


base = torch.randn(())
temp_tensor = torch.quantize_per_tensor(base, 0.1, 10, torch.qint8)
temp_tensor.topk(1, -1, False, True)