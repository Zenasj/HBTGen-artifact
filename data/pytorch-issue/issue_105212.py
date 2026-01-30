import torch

@torch.compile(mode='max-autotune')
def func(buf):
    res = torch.matmul(buf, buf)
    return res

buf = torch.randn(12, 128, 128, device='cuda')
res = func(buf)
print('res', res)