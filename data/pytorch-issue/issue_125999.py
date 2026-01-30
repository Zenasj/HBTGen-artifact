import torch

@torch.compile
def func(a, b):
    max_scale = torch.max(a, b)
    min_scale = torch.min(a, b)
    new_scale = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
    a.copy_(new_scale)

a = torch.randn(1, 32, 2, 1026).cuda()
b = torch.randn(1, 32, 1026).cuda()
print(a)
func(a[..., 1, :], b)
print(a)