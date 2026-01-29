# torch.rand(B, T, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, n_fft=16, hop_length=10):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        # Compute STFT
        stft = torch.stft(x, self.n_fft, self.hop_length, return_complex=True)
        # Reconstruct audio using ISTFT with explicit length to match input
        return torch.istft(stft, self.n_fft, self.hop_length, length=x.size(-1))

def my_model_function():
    # Initialize with parameters matching the issue's example
    return MyModel(n_fft=16, hop_length=10)

def GetInput():
    # Batch size from original test (64*512), time length 50
    B = 64 * 512
    T = 50
    return torch.rand(B, T, dtype=torch.float32)

