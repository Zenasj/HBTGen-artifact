import torch

x = torch.randn(65536, 32768, device='cuda')
print(x)