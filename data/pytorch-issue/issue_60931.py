# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn
import torch.fft as fft

class MyModel(nn.Module):
    def __init__(self, c, h, w):
        super(MyModel, self).__init__()
        real_matrix = torch.empty(c, h, w)
        imaginary_matrix = torch.empty(c, h, w)
        torch.nn.init.kaiming_normal_(real_matrix)
        torch.nn.init.kaiming_normal_(imaginary_matrix)
        cplx_matrix = torch.complex(real_matrix, imaginary_matrix)
        self.weight = torch.nn.Parameter(cplx_matrix, requires_grad=True)

    def forward(self, x):
        out = fft.fft2(x)
        out_real = torch.view_as_real(out)
        weight_real = torch.view_as_real(self.weight)
        out_multiplied = out_real * weight_real
        out_complex = torch.view_as_complex(out_multiplied)
        return fft.ifft2(out_complex).real

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(c=3, h=256, w=256)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 1  # Example batch size
    channels = 3
    height = 256
    width = 256
    return torch.rand(batch_size, channels, height, width, dtype=torch.float32)

