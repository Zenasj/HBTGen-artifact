import torch

a = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.float)
b = torch.quantize_per_tensor(a, 0.1, 10, torch.quint8)

torch.use_deterministic_algorithms(True)
b.resize_((10,))