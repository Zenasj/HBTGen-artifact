# torch.rand(2, 2, 65, dtype=torch.complex32, device='cuda')  # Inferred input shape and dtype

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, rtol=0.04, atol=0.04):
        super().__init__()
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        # Compute FFT for both dtypes
        out_complex32 = torch.fft.hfftn(x)
        out_complex64 = torch.fft.hfftn(x.to(torch.complex64))
        
        # Calculate absolute difference and threshold
        diff = torch.abs(out_complex32 - out_complex64)
        threshold = self.atol + self.rtol * torch.abs(out_complex64)
        
        # Return boolean tensor indicating if all elements meet tolerance
        return torch.all(diff < threshold)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((2, 2, 65), dtype=torch.complex32, device='cuda')

