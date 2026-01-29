# torch.rand(B, L, dtype=torch.float32)  # B=batch, L=input length (e.g., 1024)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sftt = nn.Conv1d(1, 258, 256, bias=False, stride=80)

    def forward(self, x):
        pad = 5
        padded_signal = F.pad(x, (0, pad), mode='constant', value=0)
        stft_res = self.sftt(padded_signal.unsqueeze(dim=1))
        real_squared, imag_squared = (stft_res * stft_res).split(129, dim=1)
        return real_squared  # Return only the real part as output tensor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 1024, dtype=torch.float32)  # Matches input shape (B=2, L=1024)

