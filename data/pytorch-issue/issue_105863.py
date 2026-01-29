# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.return_input = True  # Controls output branching

    def forward(self, x):
        ret = self.layers(x)[0, 0]  # Scalar output from first element
        if self.return_input:
            return ret, x  # Return tuple with input tensor
        else:
            return ret  # Return scalar only

def my_model_function():
    return MyModel()  # Returns initialized model instance

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)  # Matches input shape (B=1, C=3, H=32, W=32)

