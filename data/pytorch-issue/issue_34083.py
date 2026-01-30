import torch

x = torch.randn(200, 512, 28, 28, device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)
y = torch.max_pool2d(x, 2, 2, 0, 1, False)