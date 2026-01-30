import torch

print(torch.backends.cuda.cufft_plan_cache.max_size)

n = 5999
a = torch.randn(n, n, device='cuda', dtype=torch.complex64)
res1 = torch.fft.fft2(a)
torch.zeros_like(a)
res2 = torch.fft.fft2(a)
print(res1.abs().max(), res2.abs().max())