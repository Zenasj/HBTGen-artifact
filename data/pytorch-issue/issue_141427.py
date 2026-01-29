# torch.rand(5, 3, 6, dtype=torch.complex128)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Apply FFT along dimension 1 (matches issue's example parameters)
        fft_result = torch.fft.fft(x, dim=1, norm='forward')
        # Apply conjugate_physical (matches stride-preserved operation)
        conj_result = torch.conj_physical(fft_result)
        return conj_result

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape and dtype from issue's test examples
    return torch.rand(5, 3, 6, dtype=torch.complex128)

