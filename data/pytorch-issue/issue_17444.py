import torch

torch._C._set_cuda_deterministic(2) # 2=Error 1=Warn 0=Nothing
# not very useful, but show what it does:
torch.grid_sampler_2d(torch.randn(1,1,5,5, requires_grad=True, device='cuda'), torch.randn(1,1,5,2, requires_grad=True, device='cuda'), 0, 0).sum().backward()