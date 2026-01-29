# torch.rand(B, T, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        # Extract real part (first channel) of the complex input representation
        real_input = x[:, :, 0]
        # Apply STFT to real part (since current torch.stft doesn't support complex inputs)
        return torch.stft(
            real_input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )

def my_model_function():
    return MyModel(n_fft=16, hop_length=8)

def GetInput():
    B = 5
    T = 100
    return torch.rand(B, T, 2, dtype=torch.float32)

