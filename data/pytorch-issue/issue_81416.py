import numpy as np

import torch

def allclose(a, b, rtol, atol):
    return torch.all(torch.abs(a-b) < atol + rtol * torch.abs(b)).item()

failed = 0
N = 100
for i in range(N):
    inp = torch.rand((2, 2, 65), dtype=torch.complex32, device ='cuda')
    out_complex32 = torch.fft.hfftn(inp)
    out_complex64 = torch.fft.hfftn(inp.to(torch.complex64))
    
    failed = failed + int(allclose(out_complex32, out_complex64, rtol=0.04, atol=0.04))

print("test failed {} times out of {} times".format(failed, N))

failed = failed + int(allclose(out_complex32, out_complex64, rtol=0.04, atol=0.04))