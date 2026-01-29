import torch
from torch import nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # This forward replicates the FFT operation that triggers the cuFFT error
        return torch.fft.fft2(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor (B=1, C=1, H=60, W=60) matching the issue's input dimensions
    # Using CUDA device as the error occurs on GPU
    return torch.rand(1, 1, 60, 60, dtype=torch.float32, device="cuda")

