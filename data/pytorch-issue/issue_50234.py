# torch.rand(100, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Use center=False to avoid reflection padding error for complex tensors
        return torch.stft(x, n_fft=10, center=False, return_complex=True)

def my_model_function():
    return MyModel()

def GetInput():
    # Match the input shape and dtype from the issue's reproduction code
    return torch.rand(100, dtype=torch.complex64)

