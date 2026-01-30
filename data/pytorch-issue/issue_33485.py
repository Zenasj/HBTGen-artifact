import torch

3
size = 64
wn = torch.rand(size, size, size, dtype = torch.half, device = "cuda")
assert torch.isfinite(wn).all()
wn_freq = torch.rfft(wn, 3, onesided = True)
assert torch.isfinite(wn_freq).all() # <- FAILS