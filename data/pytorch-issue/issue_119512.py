# torch.rand(31, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Cast input to float and unsqueeze dimensions
        x = x.float()
        x = x.unsqueeze(-1)  # Shape becomes [31, 1]
        x = x.unsqueeze(0)   # Shape becomes [1, 31, 1]
        
        # Apply FFT along the second dimension (axis=1)
        fft_result = torch.fft.fft(x, dim=1)
        
        # Convert complex to real/imaginary tensor and squeeze dimensions
        fft_real_imag = torch.view_as_real(fft_result)  # Shape [1, 31, 1, 2]
        after_squeeze0 = fft_real_imag.squeeze(0)       # Shape [31, 1, 2]
        final = after_squeeze0.squeeze(1)               # Final shape [31, 2]
        
        return final

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random boolean tensor of shape [31]
    return torch.randint(0, 2, (31,), dtype=torch.bool)

