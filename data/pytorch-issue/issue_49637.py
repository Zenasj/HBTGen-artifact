# torch.rand(B, C, H, W, dtype=torch.complex64)  # Inferred input shape for complex data

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, spectrum, use_old_irfft=False):
        if use_old_irfft:
            recovered = torch.irfft(spectrum, 3, normalized=True, signal_sizes=spectrum.shape[-3:])
        else:
            spectrum_complex = torch.complex(spectrum[..., 0], spectrum[..., 1])
            recovered = torch.fft.irfftn(spectrum_complex, s=spectrum.shape[-3:], dim=(-3, -2, -1), norm='ortho')
        return recovered

def my_model_function():
    return MyModel()

def GetInput():
    torch.manual_seed(12345)
    B, C, D, H, W, _ = 1, 3, 10, 10, 10, 2
    spectrum = torch.randn(B, C, D, H, W, 2)
    return spectrum

