import torch

torch.lerp(torch.randn(10).cuda(), torch.randn(10).cuda(), torch.tensor(0.5)) # this doesn't work
# torch.compile(torch.lerp)(torch.randn(10).cuda(), torch.randn(10).cuda(), torch.tensor(0.5))  # this works

torch.add(torch.randn(10).cuda(), torch.randn(10).cuda(), alpha=torch.tensor(0.5))