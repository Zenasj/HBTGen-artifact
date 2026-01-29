# torch.rand(B, L, dtype=torch.float32)
import torch
import torch.nn as nn

class STFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 320
        self.hop_length = 160
        self.register_buffer('window', torch.hann_window(self.n_fft))
    
    def forward(self, x):
        return torch.stft(x, self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=False)

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 320
        self.hop_length = 160
        self.register_buffer('window', torch.hann_window(self.n_fft))
        self.conv = nn.Conv1d(1, 1, self.n_fft, stride=self.hop_length, bias=False)
        self.conv.weight.data = self.window.view(1, 1, self.n_fft)
        self.conv.weight.requires_grad_(False)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        return self.conv(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_model = STFTModel()
        self.conv_model = ConvModel()
    
    def forward(self, x):
        stft_out = self.stft_model(x)
        conv_out = self.conv_model(x)
        # Check if outputs are different in shape or value beyond tolerance
        shape_mismatch = stft_out.shape != conv_out.shape
        value_mismatch = not torch.allclose(stft_out, conv_out, atol=1e-5) if not shape_mismatch else False
        return torch.tensor([shape_mismatch or value_mismatch], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 512, dtype=torch.float32)

