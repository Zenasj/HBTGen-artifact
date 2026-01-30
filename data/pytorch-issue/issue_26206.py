import torch

torch.multinomial(torch.rand(2,3, device='cuda'), 5, True, out=torch.randn(2,3, device='cuda').long())