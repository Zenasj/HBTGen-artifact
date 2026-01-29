# torch.rand(5, 6, 7, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Perform FFT using the internal _fft_c2c function
        out = torch._fft_c2c(x, [0], 2, False)
        # Expected stride for CUDA/MKL (from issue description)
        expected_stride = (1, 35, 5)
        # Compare actual stride against expected
        return torch.tensor(out.stride()) == torch.tensor(expected_stride)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 6, 7, dtype=torch.complex64)

