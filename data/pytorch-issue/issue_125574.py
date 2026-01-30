import torch

def fn(a):
    idx = torch.arange(a.size(0), device=a.device)
    padded_idx = torch.constant_pad_nd(idx, (1050, 0))
    padded_idx = torch.where(padded_idx >= 0, padded_idx, padded_idx)
    return a[padded_idx]

a = torch.randn(1024, device="cuda")
cfn = torch.compile(fn)
cfn(a)

tl.device_assert((0 <= tmp9) & (tmp9 < 1024), "index out of bounds: 0 <= tmp9 < 1024")