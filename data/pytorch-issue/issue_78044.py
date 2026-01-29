# torch.rand(B, L, dtype=torch.float32)  # B=batch size, L=input length (e.g., 16000)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Force to CPU to avoid MPS FFT/complex dtype limitations
        x_cpu = x.to('cpu')
        y = torch.fft.rfft(x_cpu)
        return y.abs()

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a CPU tensor compatible with MyModel
    return torch.randn(1, 16000, dtype=torch.float32)

