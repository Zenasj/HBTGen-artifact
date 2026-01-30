import torch
device = 'cuda'
x = torch.randn(50000, 1, dtype=torch.float32)      # 50k * 4 bytes = 200 KB
expected_cpu = torch.pdist(x, p=2)                  # ~1250M * 4 bytes = 5 GB on CPU
actual_gpu = torch.pdist(x.to(device), p=2)         # 5 GB on GPU