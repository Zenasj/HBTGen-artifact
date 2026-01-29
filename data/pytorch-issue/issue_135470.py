# torch.rand(1, 32, 32, dtype=torch.float32)  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, sample_image: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(input=sample_image)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 32, 32, dtype=torch.float32)

