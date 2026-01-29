# torch.rand(2, 3, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normal = torch.distributions.normal.Normal(0, 1)

    def forward(self, x):
        return self.normal.sample(x.shape)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(2, 3)

