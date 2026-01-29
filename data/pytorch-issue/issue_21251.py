# torch.rand(B, 2, 5, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # SpectralNorm applied to both conv layers as in the issue's example
        self.a = nn.utils.spectral_norm(nn.Conv2d(2, 4, 3))
        self.b = nn.utils.spectral_norm(nn.Conv2d(4, 4, 3))  # Adjusted in_channels for forward compatibility

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape requirements for the conv layers (C=2)
    return torch.rand(1, 2, 5, 5, dtype=torch.float32)

