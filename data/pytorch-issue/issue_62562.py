import torch
import torch.nn as nn

torch.randn(10, 15, device='cuda') @ torch.randn(14, 32, device='cuda')

torch.nn.Linear(16, 32).cuda()(torch.randn(256, 17, device='cuda'))

torch.randn(10, 15, device='cuda').T @ torch.randn(9, 32, device='cuda')