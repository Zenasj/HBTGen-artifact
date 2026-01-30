import torch
a = torch.randn(1, 4, 4).cuda()

# throws error (see below)
torch.norm(a, dim=[1, 2], keepdim=True)

# works
torch.norm(a.cpu(), dim=[1, 2], keepdim=True)