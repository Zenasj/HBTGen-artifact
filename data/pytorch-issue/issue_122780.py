# torch.rand(B, L, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute magnitude via return_complex=False (real/imag components)
        y1 = torch.stft(x, n_fft=256, hop_length=160, return_complex=False)
        mag1 = torch.sqrt(torch.sum(y1 ** 2, dim=-1))
        
        # Compute magnitude via return_complex=True and .abs() (hypot implementation)
        y2 = torch.stft(x, n_fft=256, hop_length=160, return_complex=True)
        mag2 = y2.abs()
        
        # Return maximum absolute difference between the two methods
        return torch.max(torch.abs(mag2 - mag1))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 16000, dtype=torch.float)

