import torch

base = torch.randn(1)
temp_tensor = torch.quantize_per_tensor(base, 0.1, 10, torch.quint2x4)
torch.quantized_max_pool1d(temp_tensor, [], [0,0], [0,0], [0,0,1], True)